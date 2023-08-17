from typing import Any

import openai
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import QueryType

from approaches.approach import AskApproach
from core.messagebuilder import MessageBuilder
from text import nonewlines


class RetrieveThenReadApproach(AskApproach):
    """
    Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.
    """

    system_chat_template = \
    "You are an intelligent assistant helping BSH company customers with their home appliance questions, including inquiries about purchasing new products, features, configurations, and troubleshooting. " \
    "Use 'you' to refer to the individual asking the questions even if they ask with 'I'. " \
    "Answer the following question using only the data provided in the sources below. " \
    "For tabular information, return it as an HTML table. Do not return markdown format. " \
    "Each source has a name followed by a colon and the actual information; always include the source name for each fact you use in the response. " \
    "If you cannot answer using the sources below, say you don't know. Use the below example to answer. " \
    "The name of the source follows a special format: [model_number]_[document_language]-[page_number].pdf. " \
    "Use this information from source name as well, especially if someone is asking a question about a specific model. " \
    "You can mention in your answer that I found the information on page X of the manual for model Y. " \
    "Make sure to add links to the manuals in the answer in the following format: '[WGB256090_EN-54.pdf][SMS8YCI03E_EN-24.pdf]' and not in '[WGB256090_EN-54.pdf, SMS8YCI03E_EN-24.pdf]' " \

    #shots/sample conversation
    question = """

    'What is the warranty period for the Bosch washing machine model WGB256090?'

    Sources:
    WGB256090_EN-54.pdf: The warrany for the washing machine model WGB256090 is 2 years in the US and 1 year in the EU.
    SMS8YCI03E_EN-54.pdf: This offer is valid for three years from the date of purchase or at least as long as we offer support and spare parts for the relevant appliance.
    WGB256090_EN-57.pdf: BSH offers extended warranties for some models at an additional cost. Please contact your retailer for more information.
    """
    answer = "The warranty period for the BSH washing machine model WGB256090 is 2 years in the US and 1 year in the EU. [WGB256090_EN-54.pdf]"

    def __init__(self, search_client: SearchClient, openai_deployment: str, chatgpt_model: str, embedding_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.openai_deployment = openai_deployment
        self.chatgpt_model = chatgpt_model
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    async def run(self, q: str, overrides: dict[str, Any]) -> Any:
        has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_vector = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_captions = True if overrides.get("semantic_captions") and has_text else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

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
        message_builder.append_message('assistant', self.answer)
        message_builder.append_message('user', self.question)

        messages = message_builder.messages
        chat_completion = await openai.ChatCompletion.acreate(
            deployment_id=self.openai_deployment,
            model=self.chatgpt_model,
            messages=messages,
            temperature=overrides.get("temperature") or 0.3,
            max_tokens=1024,
            n=1)

        return {"data_points": results, "answer": chat_completion.choices[0].message.content, "thoughts": f"Question:<br>{query_text}<br><br>Prompt:<br>" + '\n\n'.join([str(message) for message in messages])}
