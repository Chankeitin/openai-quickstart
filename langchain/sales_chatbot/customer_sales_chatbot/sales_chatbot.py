import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def initialize_sales_bot(vector_store_dir: str="apple_customer_sale"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_base="https://api.xiaoai.plus/v1")
    
    global SALES_BOT    
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"]:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    #当向量数据库中没有合适答案时，使用LLM+prompt提示词来回答客户问题
    if len(ans["source_documents"]) == 0:
        template = """
                之前的对话内容是{history}， 客戶的最新回答是：{question}
                作为一个专业的Apple公司顾问，请给出更专业， 更自然的回复
                """
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)

        prompt = PromptTemplate(template=template, input_variables=["history", "question"])
        chain = LLMChain(llm=llm, prompt=prompt)

        response = chain.run(history=history, question=message)
        print(f"[template response] {response}")
        return response
        # 否则输出套路话术
    else:
        return "请您稍候片刻，这个问题我需要向我的上级进行核实，以确保提供给您最准确的信息。"
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="苹果官方旗舰店销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="127.0.0.1")

if __name__ == "__main__":
    # 初始化苹果官方旗舰店销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
