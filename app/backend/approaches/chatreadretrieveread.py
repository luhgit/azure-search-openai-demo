import re
import json
from typing import Any
import openai
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import QueryType

from approaches.approach import ChatApproach
from core.messagebuilder import MessageBuilder
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

    system_message_chat_conversation = """You are a customer service assistant for BSH company, helping customers with their home appliance questions, including inquiries about purchasing new products, features, configurations, and troubleshooting.
Start answering thanking the user for their question. Respond in a slightly informal, and helpful tone, with a brief and clear answers. 
Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know without referring to the sources. 
Do not generate answers that don't use the sources below and avoid to just cite the source without answering the question. 
If asking a clarifying question to the user would help, ask the question. 
For tabular information, return it as an HTML table. Do not return markdown format. 
If the question is not in English, answer in the language used in the question. 
Each source has a name followed by a colon and the actual information; always include the source name for each fact you use in the response without referring to the sources. 
For example, if the question is 'What is the capacity of this washing machine?' and one of the information sources says 'WGB256090_EN-54.pdf: the capacity is 5kg', then answer with 'The capacity is 5kg [WGB256090_EN-54.pdf]'. 
If there are multiple sources, cite each one in their own square brackets. For example, use '[WGB256090_EN-54.pdf][SMS8YCI03E_EN-24.pdf]' and not in '[WGB256090_EN-54.pdf, SMS8YCI03E_EN-24.pdf]'. 
The name of the source follows a special format: <model_number>_<document_language>-<page_number>.pdf. 
You can Use this information from source name, especially if someone is asking a question about a specific model.
{follow_up_questions_prompt}
{injected_prompt}
"""

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

    filter_prompt_template = """Based on the most recent user message in the conversation history below, accurately identify the language of the new message. 
If the message is in English, return "en-us". 
If the message is in German, return "de-de". 
If you cannot determine the language, return "unknown".

For example:
- For the message "Hello, how are you?", the answer is "en-us".
- For the message "Hallo, wie geht es Ihnen?", the answer is "de-de".

Ensure you return the correct language code enclosed in quotes, like 'en-us' or 'de-de'.
"""

#     query_prompt_template = """Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base about BSH company's home appliances, including buying guides, features, configurations, and troubleshooting.
# Identify the language of the new question and provide it in the format like 'en' for English and 'de' for German.
# Generate a search query based on the conversation including the new question. If you cannot generate a search query, return just the number 0.
# Identify and extract any mentioned product ID, model number, ENR number, serial number, WIP number, or any other identifier based on the entire conversation history. If you cannot find one then just return None.
# This number represents a unique product identifier. Use this identifier to generate a search query based on the latest user question.
# Do not include cited source filenames and document names e.g manual.pdf or catalog.pdf in the search query terms.
# Do not include any text inside [] or <<>> in the search query terms.
# Do not include any special characters like '+'.
# Return the result in the following JSON format: 
# {
#   "query": "[Your generated query]",
#   "language": "[Detected language]",
#   "product_id": "[Extracted product ID]"
# }
# """

#     query_prompt_template = """Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base about BSH company's home appliances, including buying guides, features, configurations, and troubleshooting.
# Identify the language of the most recent user message and provide it in the format like 'en' for English and 'de' for German.

# Most recent user message: '{recent_message}'

# Generate a search query based on the conversation history for the most recent user message question. If you cannot generate a search query, return just the number 0.
# Additionally, Identify and extract any mentioned product ID, model number, ENR number, serial number, WIP number, or any other identifier based on the entire conversation history. If you cannot find one then just return None.

# Conversation history: {history_content}

# This number represents a unique product identifier. Use this identifier to generate a search query based on the latest user question.
# Do not include cited source filenames and document names e.g manual.pdf or catalog.pdf in the search query terms.
# Do not include any text inside [] or <<>> in the search query terms.
# Do not include any special characters like '+'.

# Return the result in the following JSON format: 
# {{
#   "query": "[Your generated query]",
#   "language": "[Detected language]",
#   "product_id": "[Extracted product ID]"
# }}
# """

    query_prompt_few_shots = [
        {'role' : USER, 'content' : 'how to load the washing machine?' },
        {'role' : ASSISTANT, 'content' : 'Show the procedure to load a washing machine' },
        {'role' : USER, 'content' : 'Does my washing machine has wifi?' },
        {'role' : ASSISTANT, 'content' : 'Check for the wifi feature on the specified washing machine' }
    ]

    filter_prompt_few_shots = [
        {'role' : USER, 'content' : 'how to load the washing machine?' },
        {'role' : ASSISTANT, 'content' : 'en-us'}, 
        {'role' : USER, 'content' : 'Gibt es Wifi auf meine Waschmachine mit produkt nummer WGB256090?' },
        {'role' : ASSISTANT, 'content' : 'de-de'},
        {'role' : USER, 'content' : 'what are the available programms for SMS6TCI00E washing machine?' },
        {'role' : ASSISTANT, 'content' : 'en-us'},
    ]

    def __init__(self, search_client: SearchClient, chatgpt_deployment: str, chatgpt_model: str, embedding_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.chatgpt_model = chatgpt_model
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.chatgpt_token_limit = get_token_limit(chatgpt_model)

    async def run(self, history: list[dict[str, str]], overrides: dict[str, Any]) -> Any:
        has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_vector = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_captions = True if overrides.get("semantic_captions") and has_text else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        user_q = 'Generate search query for: ' + history[-1]["user"]

        print("prompt for query generation: " + user_q + "\n")

        # STEP 1: Generate an optimized keyword search query based on the chat history and the new question
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
        
        filter_q = 'Detect the language for: ' + history[-1]["user"]

        # STEP 2: Gnerate a language filter based on the chat history and the new question
        messages_filter = self.get_messages_from_history(
            self.filter_prompt_template,
            self.chatgpt_model,
            history,
            filter_q,
            self.filter_prompt_few_shots,
            self.chatgpt_token_limit - len(filter_q)
            )

        chat_completion_filter = await openai.ChatCompletion.acreate(
            deployment_id=self.chatgpt_deployment,
            model=self.chatgpt_model,
            messages=messages_filter,
            temperature=0.0,
            max_tokens=32,
            n=1)
        
        filter_content = chat_completion_filter.choices[0].message.content

        print("Message from chat history for filter generation: " + str(messages_filter))
        print("Generated filter: " + filter_content + "\n")

        language_code = filter_content
        if language_code not in ['en-us', 'de-de']:
            language_code = 'en-us'
        language_filter = f"language eq '{language_code}'"

        if filter:
            filter = f"{filter} and {language_filter}"
        else:
            filter = language_filter
        
        # response_content = chat_completion.choices[0].message.content
        # product_id = None

        # try:
        #     # Extracting query and language from the response
        #     response_json = json.loads(response_content)
        #     print('LLM output', response_json)

        #     # Check if the response is 0
        #     if response_json == 0:
        #         raise ValueError("Query generation failed")  # This will be caught by the except block below

        #     query_text = response_json["query"]
        #     language_code = response_json["language"]

        #     if query_text == 0:
        #         query_text = history[-1]["user"]

        #     # Extract product_id if available in the response
        #     if "product_id" in response_json and response_json["product_id"]:
        #         product_id = response_json["product_id"]

        # except (json.JSONDecodeError, KeyError, ValueError):
        #     # Handle edge cases where the response is not as expected
        #     query_text = history[-1]["user"]
        #     language_code = "en"  # Default to English if not specified

        # # Define language mapping
        # language_mapping = {
        #     "en": "en-us",
        #     "de": "de-de"
        # }

        # # If the detected language is neither "en" nor "de", default to "en"
        # if language_code not in language_mapping:
        #     language_code = "en"

        # # Convert the language code to the desired format
        # language_code_with_country = language_mapping[language_code]

        # # Constructing the language filter
        # language_filter = f"language eq '{language_code_with_country}'"
        
        # # If product_id is still None, attempt to extract it from the conversation history
        # if product_id is None:
        #     for message in reversed(history):
        #         match = re.search(r'\b([A-Za-z]{3}[0-9][0-9a-zA-Z]{4,6})\b', message.get('user', ''))
        #         if match:
        #             product_id = match.group(1).upper()  # Capitalize the product ID
        #             break

        # print("language code: " + language_code + "\n")
        # print(f"product id: {product_id}")

        # # Constructing the product_id filter if available
        # product_filter = ""
        # if product_id:
        #     product_filter = f"product_id eq '{product_id}'"

        # print(f"Intial filter: {filter}")
        # # Combine the filters
        # if filter:
        #     filter = f"{filter} and {language_filter}"
        #     if product_filter:
        #         filter = f"{filter} and {product_filter}"
        # else:
        #     filter = language_filter
        #     if product_filter:
        #         filter = f"{filter} and {product_filter}"

        # print("language filter: " + language_filter + "\n")
        # print("product filter: " + product_filter + "\n")

        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query

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

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history

        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = overrides.get("prompt_override")
        if prompt_override is None:
            system_message = self.system_message_chat_conversation.format(injected_prompt="", follow_up_questions_prompt=follow_up_questions_prompt)
        elif prompt_override.startswith(">>>"):
            system_message = self.system_message_chat_conversation.format(injected_prompt=prompt_override[3:] + "\n", follow_up_questions_prompt=follow_up_questions_prompt)
        else:
            system_message = prompt_override.format(follow_up_questions_prompt=follow_up_questions_prompt)

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
            if bot_msg := h.get("bot"):
                message_builder.append_message(self.ASSISTANT, bot_msg, index=append_index)
            if user_msg := h.get("user"):
                message_builder.append_message(self.USER, user_msg, index=append_index)
            if message_builder.token_length > max_tokens:
                break
        messages = message_builder.messages
        return messages

    # def get_messages_from_history(self,
    #                               system_prompt: str,
    #                               model_id: str,
    #                               history: list[dict[str, str]],
    #                               user_conv: str,
    #                               few_shots: list = [],
    #                               max_tokens: int = 4096,
    #                               output_format: bool = False
    #                               ) -> list:
    #     message_builder = MessageBuilder("", model_id)

    #     # Add examples to show the chat what responses we want. 
    #     # It will try to mimic any responses and make sure they match the rules laid out in the system message.
    #     #for shot in few_shots:
    #     #    message_builder.append_message(shot.get('role'), shot.get('content'))

    #     # Construct the history part of the prompt (Exclude the most recent message for now)
    #     history_content = "\n".join([f"user: {h['user']}" if 'user' in h else f"bot: {h['bot']}" for h in history[:-1]])


    #     print("HISTORY:", history)
    #     print("HISTORY CONTENT:", history_content)

    #     # Construct the recent message part of the prompt only for query generation
    #     if output_format:
    #         recent_message = history[-1]["user"]
    #         final_system_message = system_prompt.format(recent_message=recent_message, history_content=history_content)
    #     else:
    #         final_system_message = f"{system_prompt}\n\n{history_content}"

    #     message_builder.append_message(self.SYSTEM, final_system_message)
    #     message_builder.append_message(self.USER, user_conv)

    #     # Extract the most recent user message and emphasize it in the prompt
    #     # recent_user_message = history[-1]["user"]
    #     # emphasized_message = f"Most recent user message: \"{recent_user_message}\". Consider the entire conversation history for context:\n"
    #     # message_builder.append_message(self.SYSTEM, emphasized_message)

    #     # # Add the rest of the history
    #     # for h in reversed(history[:-1]):
    #     #     if bot_msg := h.get("bot"):
    #     #         message_builder.append_message(self.ASSISTANT, bot_msg)
    #     #     if user_msg := h.get("user"):
    #     #         message_builder.append_message(self.USER, user_msg)
    #     #     if message_builder.token_length > max_tokens:
    #     #         break

    #     messages = message_builder.messages

    #     # user_content = user_conv
    #     # append_index = len(few_shots) + 1
    #     # message_builder.append_message(self.USER, user_content, index=append_index)

    #     return messages
