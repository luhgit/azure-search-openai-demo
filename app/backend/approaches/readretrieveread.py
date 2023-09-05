from typing import Any

import openai
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import QueryType
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.callbacks.manager import CallbackManager, Callbacks
from langchain.chains import LLMChain
from langchain.llms.openai import AzureOpenAI

from approaches.approach import AskApproach
from langchainadapters import HtmlCallbackHandler
from lookuptool import CsvLookupTool
from text import nonewlines


class ReadRetrieveReadApproach(AskApproach):
    """
    Attempt to answer questions by iteratively evaluating the question to see what information is missing, and once all information
    is present then formulate an answer. Each iteration consists of two parts:
     1. use GPT to see if we need more information
     2. if more data is needed, use the requested "tool" to retrieve it.
    The last call to GPT answers the actual question.
    This is inspired by the MKRL paper[1] and applied here using the implementation in Langchain.

    [1] E. Karpas, et al. arXiv:2205.00445
    """
    #"You can mention in your answer that I found the information on page X of the manual for model Y. " Removing this to avoid expressions like "according to the source, ..."
    template_prefix = \
    "You are a customer service assistant for BSH company, helping customers with their home appliance questions, including inquiries about purchasing new products, features, configurations, and troubleshooting." \
    "Start answering thanking the user for their question. Respond in a slightly informal, and helpful tone, with a brief and clear answers. " \
    "Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know without referring to the sources. " \
    "Do not generate answers that don't use the sources below. " \
    "If asking a clarifying question to the user would help, ask the question. " \
    "For tabular information, return it as an HTML table. Do not return markdown format. " \
    "If the question is not in English, answer in the language used in the question. " \
    "Each source has a name followed by a colon and the actual information; always include the source name for each fact you use in the response without referring to the sources. " \
    "For example, if the question is 'What is the capacity of this washing machine?' and one of the information sources says 'WGB256090_EN-54.pdf: the capacity is 5kg', then answer with 'The capacity is 5kg [WGB256090_EN-54.pdf]'. " \
    "If there are multiple sources, cite each one in their own square brackets. For example, use '[WGB256090_EN-54.pdf][SMS8YCI03E_EN-24.pdf]' and not in '[WGB256090_EN-54.pdf, SMS8YCI03E_EN-24.pdf]'. " \
    "The name of the source follows a special format: <model_number>_<document_language>-<page_number>.pdf. " \
    "You can Use this information from source name, especially if someone is asking a question about a specific model. " \
    "Never quote tool names as sources. " \
    "\n\nYou can access to the following tools:"

    template_suffix = """
    Begin!

    Question: {input}

    Thought: {agent_scratchpad}"""

    CognitiveSearchToolDescription = "useful for searching the BSH homes appliances information such as operating manuals for trouibleshooting, technical data, setup guide etc."

    def __init__(self, search_client: SearchClient, openai_deployment: str, embedding_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.openai_deployment = openai_deployment
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    async def retrieve(self, query_text: str, overrides: dict[str, Any]) -> Any:
        has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_vector = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_captions = True if overrides.get("semantic_captions") and has_text else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        # If retrieval mode includes vectors, compute an embedding for the query
        if has_vector:
            query_vector = (await openai.Embedding.acreate(engine=self.embedding_deployment, input=query_text))["data"][0]["embedding"]
        else:
            query_vector = None

        # Only keep the text query if the retrieval mode uses text, otherwise drop it
        if not has_text:
            query_text = ""

        # Use semantic ranker if requested and if retrieval mode is text or hybrid (vectors + text)
        if overrides.get("semantic_ranker") and has_text:
            r = await self.search_client.search(query_text,
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC,
                                          query_language="en-us",
                                          query_speller="lexicon",
                                          semantic_configuration_name="default",
                                          top = top,
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
            results = [doc[self.sourcepage_field] + ":" + nonewlines(" -.- ".join([c.text for c in doc['@search.captions']])) async for doc in r]
        else:
            results = [doc[self.sourcepage_field] + ":" + nonewlines(doc[self.content_field][:250]) async for doc in r]
        content = "\n".join(results)
        return results, content

    async def run(self, q: str, overrides: dict[str, Any]) -> Any:

        retrieve_results = None
        async def retrieve_and_store(q: str) -> Any:
            nonlocal retrieve_results
            retrieve_results, content = await self.retrieve(q, overrides)
            return content

        # Use to capture thought process during iterations
        cb_handler = HtmlCallbackHandler()
        cb_manager = CallbackManager(handlers=[cb_handler])

        acs_tool = Tool(name="CognitiveSearch",
                        func=lambda _: 'Not implemented',
                        coroutine=retrieve_and_store,
                        description=self.CognitiveSearchToolDescription,
                        callbacks=cb_manager)
        employee_tool = EmployeeInfoTool("Employee1", callbacks=cb_manager)
        tools = [acs_tool, employee_tool]

        prompt = ZeroShotAgent.create_prompt(
            tools=tools,
            prefix=overrides.get("prompt_template_prefix") or self.template_prefix,
            suffix=overrides.get("prompt_template_suffix") or self.template_suffix,
            input_variables = ["input", "agent_scratchpad"])
        llm = AzureOpenAI(deployment_name=self.openai_deployment, temperature=overrides.get("temperature") or 0.3, openai_api_key=openai.api_key)
        chain = LLMChain(llm = llm, prompt = prompt)
        agent_exec = AgentExecutor.from_agent_and_tools(
            agent = ZeroShotAgent(llm_chain = chain),
            tools = tools,
            verbose = True,
            callback_manager = cb_manager)
        result = await agent_exec.arun(q)

        # Remove references to tool names that might be confused with a citation
        result = result.replace("[CognitiveSearch]", "").replace("[Employee]", "")

        return {"data_points": retrieve_results or [], "answer": result, "thoughts": cb_handler.get_and_reset_log()}

class EmployeeInfoTool(CsvLookupTool):
    employee_name: str = ""

    def __init__(self, employee_name: str, callbacks: Callbacks = None):
        super().__init__(filename="data/employeeinfo.csv",
                         key_field="name",
                         name="Employee",
                         description="useful for searching the BSH homes appliances information such as operating manuals for trouibleshooting, technical data, setup guide etc.",
                         callbacks=callbacks)
        self.func = lambda _: 'Not implemented'
        self.coroutine = self.employee_info
        self.employee_name = employee_name

    async def employee_info(self, name: str) -> str:
        return self.lookup(name)
