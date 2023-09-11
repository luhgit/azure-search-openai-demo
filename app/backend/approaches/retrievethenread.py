from typing import Any

import openai
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import QueryType
import re 

from core.modelhelper import get_token_limit
from approaches.approach import AskApproach
from core.messagebuilder import MessageBuilder
from core.language_detector import detect_language
from text import nonewlines


class RetrieveThenReadApproach(AskApproach):
    """
    Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.
    """

    system_chat_template = """You are a customer service assistant for BSH company, helping customers with their home appliance questions, including inquiries about purchasing new products, features, configurations, and troubleshooting.
Start answering thanking the user for their question. Respond in a slightly informal, and helpful tone, with a brief and clear answers. 
Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know without referring to the sources. 
Try to answer the question in detail and avoid to just cite the source without answering the question.
If the question is about a specific product, describe the answer in details and avoid referring to sources if possible. e.g., providing a step by step guidance in your response.
Avoid refering the user to the product manual or catalog and try to answer the question directly.
If asking a clarifying question to the user would help, ask the question. 
For tabular information, return it as an HTML table. Do not return markdown format. 
If the question is not in English, answer in the language used in the question.
Do not generate answers that don't use the sources below.
Each source has a name followed by a colon and the actual information contained within the source.
Always include the source name for each fact you use in the response. 
For example, if the question is 'What is the capacity of this washing machine?' and one of the information sources says 'WGB256090_en-us_dishwasher_product-manual-3.pdf: the capacity is 5kg', then answer with 'The capacity is 5kg [WGB256090_en-us_dishwasher_product-manual-3.pdf]'. 
Cite the only the source names that are provided to you and do not mention sources that are not known to you.
Cite the exact name of the source as provided to you and do not change the source name.
If there are multiple sources, cite each one in their own square brackets. For example, use '[WGB256090_en-us_dishwasher_prodcut-manual-54.pdf][SMS8YCI03E_en-us_dishwasher_product-manual-12.pdf]' and not in '[WGB256090_en-us_dishwasher_product-manual-54.pdf, SMS8YCI03E_en-us_dishwasher_manual-12.pdf]'.
"""

    product_filter_template = """Based on the user message accurately identify the product id.
Product ids are made of alpha-numeric characters like "SMS6TCI00E", "WUU28TA8", return ONLY the product ID. 
If the product ID isn't clear or not mentioned, return "unknown".
If the question is general and not about a specific product, return "unknown".

Ensure you return the answer in the following format e.g., 'SMD6TCX00E', 'WUU28TA8', 'unknown'.

Examples:
"user": "what is the warranty period for the Bosch washing machine model WGB256090?"
"bot": "WGB256090"

"user": 'what are the dimentions for washing machine: SMD6TCX00E?'
"bot": "SMD6TCX00E"

"user": 'how to load a washing machine?'
"bot": "unknown"
"""

    # shots/sample conversation
    last_question_example = """

    'What is the warranty period for the Bosch washing machine model WGB256090?'

    Sources:
    WGB256090_en-us_dishwasher_manual-54.pdf: The warrany for the washing machine model WGB256090 is 2 years in the US and 1 year in the EU.
    SMS8YCI03E_en-us_dishwasher_manual-54.pdf: This offer is valid for three years from the date of purchase or at least as long as we offer support and spare parts for the relevant appliance.
    WGB256090_en-us_dishwasher_manual-57.pdf: BSH offers extended warranties for some models at an additional cost. Please contact your retailer for more information.
    """
    last_answer_example = "The warranty period for the BSH washing machine model WGB256090 is 2 years in the US and 1 year in the EU. [WGB256090_en-us_dishwasher_manual-54.pdf]"

    def __init__(self, search_client: SearchClient, openai_deployment: str, chatgpt_model: str, embedding_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.openai_deployment = openai_deployment
        self.chatgpt_model = chatgpt_model
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.chatgpt_token_limit = get_token_limit(chatgpt_model)

    async def run(self, q: str, overrides: dict[str, Any]) -> Any:
        has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_vector = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_captions = True if overrides.get("semantic_captions") and has_text else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        # STEP 1: Gnerate a product filter based on the chat history and the new question
        product_filter_q = 'Detect the product id for: ' + q
        message_product_filter = MessageBuilder(self.product_filter_template, self.chatgpt_model)
        message_product_filter.append_message('user', product_filter_q)

        messages_product = message_product_filter.messages

        chat_completion_filter = await openai.ChatCompletion.acreate(
            deployment_id=self.openai_deployment,
            model=self.chatgpt_model,
            messages=messages_product,
            temperature=0.0,
            max_tokens=32,
            n=1)
        
        product_filter_content = chat_completion_filter.choices[0].message.content

        print("Message from chat history for product filter generation: " + str(messages_product))
        print("Generated product: " + product_filter_content + "\n")

        # STEP 2: Gnerate a language filter based on the last question
        language_code = detect_language(q)
        language_filter = f"language eq '{language_code}'"
        print("Generated language: " + language_filter + "\n")
        
        product_filter = None
        product_id = product_filter_content
        pattern = r'^[A-Za-z]{3}[0-9][0-9a-zA-Z]{4,8}$'
        if re.match(pattern, product_id):
            product_filter = f"product_id eq '{product_id}'"

        if filter:
            filter = f"{filter} and {language_filter}"
            if product_filter:
                filter = f"{filter} and {product_filter}"
        else:
            filter = language_filter
            if product_filter:
                filter = f"{filter} and {product_filter}"

        # STEP 4: Retrieve relevant documents from the search index with the GPT optimized query

        # If retrieval mode includes vectors, compute an embedding for the query
        if has_vector:
            query_vector = (await openai.Embedding.acreate(engine=self.embedding_deployment, input=q))["data"][0]["embedding"]
        else:
            query_vector = None

        # Only keep the text query if the retrieval mode uses text, otherwise drop it
        query_text = q if has_text else ""

        # Use semantic ranker if requested and if retrieval mode is text or hybrid (vectors + text)
        if overrides.get("semantic_ranker") and has_text:
            r = await self.search_client.search(query_text,
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC,
                                          query_language="en-us",
                                          query_speller="lexicon",
                                          semantic_configuration_name="default",
                                          top=top,
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None,
                                          vector=query_vector,
                                          top_k=50 if query_vector else None,
                                          vector_fields="embedding" if query_vector else None)
        else:
            r = await self.search_client.search(query_text,
                                          filter=filter,
                                          top=top,
                                          vector=query_vector,
                                          top_k=50 if query_vector else None,
                                          vector_fields="embedding" if query_vector else None)
        if use_semantic_captions:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) async for doc in r]
        else:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) async for doc in r]
        content = "\n".join(results)

        message_builder = MessageBuilder(overrides.get("prompt_template") or self.system_chat_template, self.chatgpt_model)

        # add user question
        user_content = q + "\n" + f"Sources:\n {content}"
        message_builder.append_message('user', user_content)

        # Add shots/samples. This helps model to mimic response and make sure they match rules laid out in system message.
        message_builder.append_message('assistant', self.last_answer_example)
        message_builder.append_message('user', self.last_question_example)

        messages = message_builder.messages
        chat_completion = await openai.ChatCompletion.acreate(
            deployment_id=self.openai_deployment,
            model=self.chatgpt_model,
            messages=messages,
            temperature=overrides.get("temperature") or 0.3,
            max_tokens=1024,
            n=1)

        return {"data_points": results, "answer": chat_completion.choices[0].message.content, "thoughts": f"Question:<br>{query_text}<br><br>Prompt:<br>" + '\n\n'.join([str(message) for message in messages])}
