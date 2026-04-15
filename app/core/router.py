from .llm_client import LLMService


class IntentRouter:
    def __init__(self, api_key, base_url):
        self.llm = LLMService(
            api_key=api_key,
            base_url=base_url,
            timeout=10.0,
            max_retries=2,
            backoff_base_seconds=0.3,
        )

    def classify(self, question: str) -> str:
        prompt = f"""你是一个任务分发器。请分析用户的输入，仅返回以下三个标签之一：
- [Chat]: 如果用户是在打招呼、闲聊、问你是谁、或者进行礼貌性回复。
- [Data_Query]: 如果用户在询问有关客户、订单、产品、员工、供应商等数据库相关的数据统计或查询。
- [Unsupported]: 如果用户要求你写代码、翻译、或者做与数据库查询无关的事情。

用户输入: "{question}"
标签: """

        result = self.llm.call(prompt, temperature=0.0, max_tokens=10)
        if not result.ok:
            return "query"  # 发生异常时默认走查询流，保证鲁棒性
        tag = result.content.strip()
        if "[Chat]" in tag:
            return "chat"
        if "[Data_Query]" in tag:
            return "query"
        return "unsupported"
