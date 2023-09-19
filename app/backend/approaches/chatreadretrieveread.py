import re
import yaml
from typing import Any
import openai
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import QueryType

from approaches.approach import ChatApproach
from core.messagebuilder import MessageBuilder
from core.language_detector import detect_language
from core.modelhelper import get_token_limit
from text import nonewlines


class ChatReadRetrieveReadApproach(ChatApproach):
    # Chat roles
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    """
    Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.
    """

    system_message_chat_conversation_withid = """"You are the customer service assistant for BSH, tasked with helping customers with their home appliance inquiries, including purchasing new products, features, configurations, and troubleshooting. Your role is to provide friendly and concise responses while ensuring customer satisfaction.
    Only brands available are Bosh and Siemens, do not answer to other brands questions.
Here's a step-by-step guide on how to assist users effectively:

Step 1: Always start by thanking the user for their question. Maintain a slightly informal, helpful tone throughout your responses.

Step 2: Use only the information available in the list of sources provided below to answer the user's question. If the question is unrelated to BSH products, inform the user that you cannot provide answers for those queries. If the sources do not contain sufficient information, state that you do not have the necessary details.

Step 3: Only generate responses based on the following sources provided; do not create answers from unknown sources. If you cannot find the answer, state that you do not have the necessary details.
Provide detailed answers, avoiding simple source citations without addressing the user's question. You can repeat what's inside the source.
Keep your answers brief and clear and human-readable. You should use bullet points if this could help.
Do not direct the user to product manuals or catalogs; aim to answer the question directly.
If you need additional information to assist the user, feel free to ask clarifying questions.
Respond in the language used by the user in their question.
For tabular information, return it as an HTML table. Do not return markdown format. 

Step 4: Remember to cite the exact source name when providing information in square brakets. Each source has a name followed by a colon and the actual information contained within the source.
Always include the source name for each fact you use in the response. 
For example, if the question is 'What is the capacity of this washing machine?' and one of the information sources says 'WGB256090_en-us_dishwasher_product-manual-3.pdf: the capacity is 5kg', then answer with 'The capacity is 5kg [WGB256090_en-us_dishwasher_product-manual-3.pdf]'. 
Cite the only the source names that are provided to you and do not mention sources that are not known to you.
Cite the exact name of the source as provided to you and do not change the source name.
If there are multiple sources, cite each one in their own square brackets. For example, use '[WGB256090_en-us_dishwasher_prodcut-manual-54.pdf][SMS8YCI03E_en-us_dishwasher_product-manual-12.pdf]' and not in '[WGB256090_en-us_dishwasher_product-manual-54.pdf, SMS8YCI03E_en-us_dishwasher_manual-12.pdf]'.

{follow_up_questions_prompt}
{injected_prompt}
"""

    system_message_chat_conversation_nodocument = """"You are the customer service assistant for BSH, tasked with helping customers with their home appliance inquiries, including purchasing new products, features, configurations, and troubleshooting. Your role is to provide friendly and concise responses while ensuring customer satisfaction.

Unfortunatly we don't have an answer for the client's question. 
Try to suggest to reformulate the answer and if he can't find a solution for his problem he can contact our support team. 
{follow_up_questions_prompt}
{injected_prompt}
"""

    system_message_chat_conversation_noid = """"You are the customer service assistant for BSH, tasked with helping customers with their home appliance inquiries, including purchasing new products, features, configurations, and troubleshooting. Your role is to provide friendly and concise responses while ensuring customer satisfaction.
    Only brands available are Bosh and Siemens, do not answer to other brands questions.

Here's a step-by-step guide on how to assist users effectively:

Step 1: Always start by thanking the user for their question. Maintain a slightly informal, helpful tone throughout your responses. 

Step 2: IMPORTANT: Always ask the user to provide the ENR number and do not answer to the question. Explain that providing the product ID is beneficial because BSH offers a range of products with varying functionalities. By having this information, you can tailor your response more accurately.
The ENR number for dishwashers can usually be found on a label, or engraved, on the top or side rim of the door, visible when the door is opened.
For washing machines the model number can usually be found on a label on the inner rim of the door, visible when the door is opened. Sometimes it can also be found behind the filter door or on the back of the appliance.

Step 3: ONLY if the user explicitly say that doesn't have the ENR, use the information available in the list of sources provided below to answer the user's question. If the question is unrelated to BSH products, inform the user that you cannot provide answers for those queries. If the sources do not contain sufficient information, state that you do not have the necessary details.

Step 4: If the user explecitly say doesn't have the ENR, only generate responses based on the following sources provided; do not create answers from unknown sources. If you cannot find the answer, state that you do not have the necessary details.
Provide detailed answers, avoiding simple source citations without addressing the user's question. You can repeat what's inside the source.
Keep your answers brief and clear and human-readable. You should use bullet points if this could help.
Do not direct the user to product manuals or catalogs; aim to answer the question directly.
If you need additional information to assist the user, feel free to ask clarifying questions.
Respond in the language used by the user in their question.

Step 5: Remember to cite the exact source name when providing information in square brakets. Each source has a name followed by a colon and the actual information contained within the source.
Always include the source name for each fact you use in the response. 
For example, if the question is 'What is the capacity of this washing machine?' and one of the information sources says 'WGB256090_en-us_dishwasher_product-manual-3.pdf: the capacity is 5kg', then answer with 'The capacity is 5kg [WGB256090_en-us_dishwasher_product-manual-3.pdf]'. 
Cite the only the source names that are provided to you and do not mention sources that are not known to you.
Cite the exact name of the source as provided to you and do not change the source name.
If there are multiple sources, cite each one in their own square brackets. For example, use '[WGB256090_en-us_dishwasher_prodcut-manual-54.pdf][SMS8YCI03E_en-us_dishwasher_product-manual-12.pdf]' and not in '[WGB256090_en-us_dishwasher_product-manual-54.pdf, SMS8YCI03E_en-us_dishwasher_manual-12.pdf]'.
{follow_up_questions_prompt}
{injected_prompt}
"""

    answer_wrong_id = """Thanks for your question. The provided ENR seems incorrect. Please verify the product ID and try again."""

    follow_up_questions_prompt_content = """Generate three very brief follow-up questions that the user would likely ask next about the home appliance they are interested in or need help with. 
Use double angle brackets to reference the questions, e.g. <<Is there a warranty on this washing machine?>>. 
Try not to repeat questions that have already been asked. 
Only generate questions and do not generate any text before or after the questions, such as 'Next Questions'
"""

    query_prompt_template = """Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base about BSH company's home appliances, including buying guides, features, configurations, and troubleshooting.
Generate a search query based on the conversation and the new question. 
Ensure that the search query is in the same language as the new question.
If the question is not in English, answer in the language used in the question.
Do not include cited source filenames and document names e.g info.txt or doc.pdf in the search query terms.
Do not include any text inside [] or <<>> in the search query terms.
Do not include any special characters like '+'.
If you cannot generate a search query, return just the number 0.
Return the query enclosed in the quotes for e.g., 'washing machine installation procedure'
"""

    product_filter_template = """Based on the entire conversation history below:
You have two task.
1: Accurately identify the product id.
Even if it's from previous messages in the conversation. 
If there are multiple product ids, return the most recent one.
Ensure that the last question is still referring to the correct product id.
Product ids are made of alpha-numeric characters like "SMS6TCI00E", "WUU28TA8". 
If the product ID isn't clear or not mentioned, try to understand if the question is about a dish-washer or a washing machine. 
If it's impossible to determine return "unknown".

2: Understand if the last user question is about a washing-machine or a dish-washer.
If it's impossible to determine return "unknown". Only possible values are "WASHING MACHINE", "DISH WASHER" and "unknown".

Ensure you return the the two answers separated by a comma without spaces: e.g., 'SMD6TCX00E,WASHING MACHINE', 'WUU28TA8,DISH WASHER', 'unknown,WASHING MACHINE', 'unknown,DISH WASHER'.
"""

    query_prompt_few_shots = [
        {'role' : USER, 'content' : 'how to load the washing machine?' },
        {'role' : ASSISTANT, 'content' : 'Show the procedure to load a washing machine' },
        {'role' : USER, 'content' : 'Does my washing machine has wifi?' },
        {'role' : ASSISTANT, 'content' : 'Check for the wifi feature on the specified washing machine' }
    ]

    product_filter_prompt_few_shots = [
    {'role' : USER, 'content' : 'Gibt es Wifi auf meine Waschmachine mit produkt nummer WGB256090?' },
    {'role' : ASSISTANT, 'content' : 'WGB256090,WASHING MACHINE'},
    {'role' : USER, 'content' : 'what are the available programms for washing machine I mentioned?' },
    {'role' : ASSISTANT, 'content' : 'WGB256090,WASHING MACHINE'},
    {'role' : USER, 'content' : 'how to load a washing machine?' },
    {'role' : ASSISTANT, 'content' : 'unknown,WASHING MACHINE'},
    {'role' : USER, 'content' : 'what are the dimentions for the dish washer: SMD6TCX00E?' },
    {'role' : ASSISTANT, 'content' : 'SMD6TCX00E,DISH WASHER'}
    ]

    def __init__(self, search_client: SearchClient, chatgpt_deployment: str, chatgpt_model: str, embedding_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.chatgpt_model = chatgpt_model
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.config = self.read_config()
        self.chatgpt_token_limit = get_token_limit(chatgpt_model)
    
    def read_config(self):
        yaml_file_path = '../config/model_config.yaml'

        # Load the YAML file
        with open(yaml_file_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config

    async def run(self, history: list[dict[str, str]], overrides: dict[str, Any]) -> Any:
        has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_vector = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_captions = True if overrides.get("semantic_captions") and has_text else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        user_q = 'Generate search query for: ' + history[-1]["user"]

        print("prompt for query generation: " + user_q + "\n")

        # STEP 1: Gnerate a product filter based on the chat history and the new question
        product_filter_q = 'Detect only the product id for: ' + history[-1]["user"]
        messages_product_filter = self.get_messages_from_history(
            self.product_filter_template,
            self.chatgpt_model,
            history,
            product_filter_q,
            self.product_filter_prompt_few_shots,
            self.chatgpt_token_limit - len(product_filter_q)
            )

        chat_completion_filter = await openai.ChatCompletion.acreate(
            deployment_id=self.chatgpt_deployment,
            model=self.chatgpt_model,
            messages=messages_product_filter,
            temperature=0.0,
            max_tokens=32,
            n=1)
        
        product_filter_content = chat_completion_filter.choices[0].message.content
        try:
            product_id, product_type = product_filter_content.split(',')
        except ValueError:
            product_id, product_type = "unknown", "unknown"
        product_type = f"product_type eq '{product_type}'" if product_type in ["WASHING MACHINE", "DISH WASHER"] else None
        print("Message from chat history for product filter generation: " + str(messages_product_filter))
        print("Generated product: " + product_filter_content + "\n")

        # If the product id is not correct, return an error message
        pattern = r'^[A-Za-z]{3}[0-9][0-9a-zA-Z]{4,8}$'

        if re.match(pattern, product_id) and product_id not in self.config["product_ids"]:
            return  {"data_points": [], 
                "answer": self.answer_wrong_id, 
                "thoughts": "the provided product id is not present in the config-model_config.yaml file"
                }
        
        # STEP 2: Genrate a language filter based on the last question
        language_code = detect_language(history[-1]["user"])
        language_filter = f"language eq '{language_code}'"
        print("Generated language: " + language_filter + "\n")

        # STEP 3: Define filter
        product_filter = None
        
        if re.match(pattern, product_id):
            product_filter = f"product_id eq '{product_id}'"

        if filter:
            filter = f"{filter} and {language_filter}"
            if product_filter:
                filter = f"{filter} and {product_filter}"
            elif product_type:
                filter = f"{filter} and {product_type}"
        else:
            filter = language_filter
            if product_filter:
                filter = f"{filter} and {product_filter}"
            elif product_type:
                filter = f"{filter} and {product_type}"
        
        # STEP 4: Generate an optimized keyword search query based on the chat history and the new question
        messages_query = self.get_messages_from_history(
            self.query_prompt_template,
            self.chatgpt_model,
            history,
            user_q,
            self.query_prompt_few_shots,
            self.chatgpt_token_limit - len(user_q)
            )

        chat_completion_query = await openai.ChatCompletion.acreate(
            deployment_id=self.chatgpt_deployment,
            model=self.chatgpt_model,
            messages=messages_query,
            temperature=0.0,
            max_tokens=32,
            n=1)
        
        query_content = chat_completion_query.choices[0].message.content
        query_text = query_content

        print("Message from chat history for query generation: " + str(messages_query) + "\n")
        print("Generated query: " + query_text)

        print(f"Final filter {filter}")

        # STEP 4: Retrieve relevant documents from the search index with the GPT optimized query

        # If retrieval mode includes vectors, compute an embedding for the query
        if has_vector:
            query_vector = (await openai.Embedding.acreate(engine=self.embedding_deployment, input=query_text))["data"][0]["embedding"]
        else:
            query_vector = None

         # Only keep the text query if the retrieval mode uses text, otherwise drop it
        if not has_text:
            query_text = None

        # Use semantic L2 reranker if requested and if retrieval mode is text or hybrid (vectors + text)
        if overrides.get("semantic_ranker") and has_text:
            r = await self.search_client.search(query_text,
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC,
                                          query_language=language_code,
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

        follow_up_questions_prompt = self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else ""

        # STEP 5: Generate a contextual and content specific answer using the search results and chat history

        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = overrides.get("prompt_override")
        if prompt_override is None:
            if product_filter:
                if len(content) > 3:
                    system_message = self.system_message_chat_conversation_withid.format(injected_prompt="", follow_up_questions_prompt=follow_up_questions_prompt)
                else:
                    print("here")
                    system_message = self.system_message_chat_conversation_nodocument.format(injected_prompt="", follow_up_questions_prompt=follow_up_questions_prompt)
            else:
                if len(content) > 3:
                    system_message = self.system_message_chat_conversation_noid.format(injected_prompt="", follow_up_questions_prompt=follow_up_questions_prompt)
                else:
                    system_message = self.system_message_chat_conversation_nodocument.format(injected_prompt="", follow_up_questions_prompt=follow_up_questions_prompt)

        messages_answer = self.get_messages_from_history(
            system_message + "\n\nSources:\n" + content,
            self.chatgpt_model,
            history,
            history[-1]["user"],
            max_tokens=self.chatgpt_token_limit)
        
        chat_completion_answer = await openai.ChatCompletion.acreate(
            deployment_id=self.chatgpt_deployment,
            model=self.chatgpt_model,
            messages=messages_answer,
            temperature=overrides.get("temperature") or 0.7,
            max_tokens=1024,
            n=1)
        
        chat_content = chat_completion_answer.choices[0].message.content
        msg_to_display = '\n\n'.join([str(message) for message in messages_answer])

        print("Message from chat history for answer generation: " + str(messages_answer) + "\n")
        print("Generated answer: " + chat_content)

        return {"data_points": results, 
                "answer": chat_content, 
                "thoughts": f"Searched for:<br>{query_text}<br><br>Conversations:<br>" + msg_to_display.replace('\n', '<br>')
                }

    def get_messages_from_history(self, system_prompt: str, model_id: str, history: list[dict[str, str]], user_conv: str, few_shots = [], max_tokens: int = 4096) -> list:
        message_builder = MessageBuilder(system_prompt, model_id)
        # Add examples to show the chat what responses we want. It will try to mimic any responses and make sure they match the rules laid out in the system message.
        for shot in few_shots[::-1]:
            message_builder.append_message(shot.get('role'), shot.get('content'))
        user_content = user_conv
        append_index = len(few_shots) + 1
        message_builder.append_message(self.USER, user_content, index=append_index)
        for h in reversed(history[:-1]):
            if message_builder.token_length > max_tokens:
                break
            if bot_msg := h.get("bot"):
                message_builder.append_message(self.ASSISTANT, bot_msg, index=append_index)
            if user_msg := h.get("user"):
                message_builder.append_message(self.USER, user_msg, index=append_index)
        messages = message_builder.messages
        return messages