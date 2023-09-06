from typing import Any
import json
import openai
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import QueryType
import re

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
Each source has a name followed by a colon and the actual information; always include the source name for each fact you use but first try to give an answer and then provide the source you are using. 
For example, if the question is 'What is the capacity of this washing machine?' and one of the information sources says 'WGB256090_EN-54.pdf: the capacity is 5kg', then answer with 'The capacity is 5kg [WGB256090_EN-54.pdf]'. 
If there are multiple sources, cite each one in their own square brackets. For example, use '[WGB256090_EN-54.pdf][SMS8YCI03E_EN-24.pdf]' and not in '[WGB256090_EN-54.pdf, SMS8YCI03E_EN-24.pdf]'. 
The name of the source follows a special format: <model_number>_<document_language>-<page_number>.pdf. 
You can Use this information from source name, especially if someone is asking a question about a specific model.
{follow_up_questions_prompt}
{injected_prompt}
"""

    system_message_chat_conversation_no_sources = """You are a customer service assistant for BSH company.
Start thanking the user for his question and please say that unfortunately you cannot anwer with the information available.
If the user is referring to a specific product id, ask to try to double check the product id.
If the question is not that clear, politely ask to reformulate it.
For example, if the question is 'what are the available programms for SMS6TCI00E washing machine?' then answer with 'Unfortunately I cannot answer to your question. Can you please double check the product id?'
If the question is 'what are the available programms for washing machine?' then answer with 'Unfortunately I cannot answer to your question. Can you please reformulate it?
"""
    follow_up_questions_prompt_content = """Generate three very brief follow-up questions that the user would likely ask next about the home appliance they are interested in or need help with. 
Use double angle brackets to reference the questions, e.g. <<Is there a warranty on this washing machine?>>. 
Try not to repeat questions that have already been asked. 
Only generate questions and do not generate any text before or after the questions, such as 'Next Questions'
"""
    
    query_prompt_template = """Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base about BSH company's home appliances, including buying guides, features, configurations, and troubleshooting.
Generate a search query based on the conversation and the new question. 
Do not include cited source filenames and document names e.g info.txt or doc.pdf in the search query terms.
Do not include any text inside [] or <<>> in the search query terms.
Do not include any special characters like '+'.
If the question is not in English, translate the question to English before generating the search query.
If you cannot generate a search query, return just the number 0.
"""
    query_prompt_few_shots = [
{'role' : USER, 'content' : 'how to load the washing machine?' },
{'role' : ASSISTANT, 'content' : 'Show the procedure to load a washing machine' },
{'role' : USER, 'content' : 'Does my washing machine has wifi?' },
{'role' : ASSISTANT, 'content' : 'Check for the wifi feature on the specified washing machine' }

]

    filter_prompt_template = """Below is a history of the conversation so far, and a new question asked by the user. 
First step: identify the language of the LAST user question and return "en-us" if it's in english and "de-de" if it's in german.
If you don't know the language, return "unknown".
Possible answers are: "en-us", "de-de", "unknown".
Second step: identify the product mentioned in the question and return the product id. 
The product id could be mentioned in the history. Be sure the last question is still referring to the product id.
If you don't know the which product the client is talking about because it's not mentioned explicitly in the question, return "unknown".
Product ids are only alpha-numeric characters like "SMS6TCI00E", "WUU28TA8", if it's not clear, return "unknown".
Possible answers are: "SMS6TCI00E", "WUU28TA8", ..., "unknown".
 
Return the two answers separated by a comma, e.g. "en-us,SMS6TCI00E".
"""

    filter_prompt_few_shots = [
{'role' : USER, 'content' : 'how to load the washing machine?' },
{'role' : ASSISTANT, 'content' : 'en-us,unknown' }, 
{'role' : USER, 'content' : 'what are the available programms for SMS6TCI00E washing machine?' },
{'role' : ASSISTANT, 'content' : 'en-us,SMS6TCI00E' }, 
{'role' : USER, 'content' : 'what are the available programms for SMD6TCX00E washing machine?' },
{'role' : ASSISTANT, 'content' : 'en-us,SMD6TCX00E' }
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
        
        ### FILTERING step definition ###
        user_q = 'User question: ' + history[-1]["user"]

        print("prompt for query generation: " + user_q + "\n")

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        messages_filtering = self.get_messages_from_history(
            self.filter_prompt_template,
            self.chatgpt_model,
            history,
            user_q,
            self.filter_prompt_few_shots,
            self.chatgpt_token_limit - len(user_q)
            )

        print("Message from chat history: " + str(messages_filtering) + "\n")

        chat_completion_filter = await openai.ChatCompletion.acreate(
            deployment_id=self.chatgpt_deployment,
            model=self.chatgpt_model,
            messages=messages_filtering,
            temperature=0.0,
            max_tokens=32,
            n=1)

        filtering_content = chat_completion_filter.choices[0].message.content
        language_code_with_country, product_query = filtering_content.split(",")
        # Constructing the language and product id filter
        
        if language_code_with_country in ["en-us", "de-de"]:
            language_filter = f"language eq '{language_code_with_country}'"
        else:
            language_filter = "language eq 'en-us'"

        pattern = r'^[A-Z]{3}[A-Z0-9]{5,9}$'
        if re.match(pattern, product_query):
            product_filter = f"product_id eq '{product_query}'"
            filter = f"{language_filter} and {product_filter}"
        else:
            filter = language_filter
        
        print("Filter: " + filter + "\n")

        #### STEP 2: Generate an optimized keyword search query based on the chat history and the last question
        user_q = 'Generate search query for: ' + history[-1]["user"]
        print("prompt for query generation: " + user_q + "\n")

        messages = self.get_messages_from_history(
            self.query_prompt_template,
            self.chatgpt_model,
            history,
            user_q,
            self.query_prompt_few_shots,
            self.chatgpt_token_limit - len(user_q)
            )

        print("Message from chat history: " + str(messages) + "\n")

        chat_completion = await openai.ChatCompletion.acreate(
            deployment_id=self.chatgpt_deployment,
            model=self.chatgpt_model,
            messages=messages,
            temperature=0.0,
            max_tokens=32,
            n=1)

        response_content = chat_completion.choices[0].message.content
        
        # STEP 3: Retrieve relevant documents from the search index with the GPT optimized query

        print("Generated query: " + response_content + "\n")

        # If retrieval mode includes vectors, compute an embedding for the query
        if has_vector:
            query_vector = (await openai.Embedding.acreate(engine=self.embedding_deployment, input=response_content))["data"][0]["embedding"]
        else:
            query_vector = None

         # Only keep the text query if the retrieval mode uses text, otherwise drop it
        if not has_text:
            response_content = None

        # Use semantic L2 reranker if requested and if retrieval mode is text or hybrid (vectors + text)
        if overrides.get("semantic_ranker") and has_text:
            r = await self.search_client.search(response_content,
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC,
                                          query_language=language_code_with_country,
                                          query_speller="lexicon",
                                          semantic_configuration_name="default",
                                          top=top,
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None,
                                          vector=query_vector,
                                          top_k=50 if query_vector else None,
                                          vector_fields="embedding" if query_vector else None)
        else:
            r = await self.search_client.search(response_content,
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
        
        print("Retrieved documents: " + content + "\n")

        follow_up_questions_prompt = self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else ""

        print("Follow up questions prompt: " + follow_up_questions_prompt + "\n")

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history

        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = overrides.get("prompt_override")
        if prompt_override is None:
            system_message = self.system_message_chat_conversation.format(injected_prompt="", follow_up_questions_prompt=follow_up_questions_prompt)
        elif prompt_override.startswith(">>>"):
            system_message = self.system_message_chat_conversation.format(injected_prompt=prompt_override[3:] + "\n", follow_up_questions_prompt=follow_up_questions_prompt)
        else:
            system_message = prompt_override.format(follow_up_questions_prompt=follow_up_questions_prompt)

        new_history = history if len(content) > 0 else [h for h in history[-1:] if h.get("user")]
        messages = self.get_messages_from_history(
            system_message + "\n\nSources:\n" + content if len(content) > 0 else self.system_message_chat_conversation_no_sources,
            self.chatgpt_model,
            new_history, 
            history[-1]["user"],
            max_tokens=self.chatgpt_token_limit)
        
        print("Message from chat history: " + str(messages) + "\n")

        chat_completion = await openai.ChatCompletion.acreate(
            deployment_id=self.chatgpt_deployment,
            model=self.chatgpt_model,
            messages=messages,
            temperature=overrides.get("temperature") or 0.7,
            max_tokens=1024,
            n=1)
        
        print("Generated answer: " + chat_completion.choices[0].message.content + "\n")

        chat_content = chat_completion.choices[0].message.content

        print("Chat content: " + chat_content + "\n")

        msg_to_display = '\n\n'.join([str(message) for message in messages])

        print("Message to display: " + msg_to_display + "\n")

        return {"data_points": results, "answer": chat_content, "thoughts": f"Searched for:<br>{response_content}<br><br>Conversations:<br>" + msg_to_display.replace('\n', '<br>')}

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
