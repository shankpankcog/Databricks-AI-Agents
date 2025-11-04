# guardrails.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from logger_config import get_logger

logger = get_logger(__name__)

def is_content_safe(llm, content: str) -> bool:
    """
    Checks if the given content is safe by asking a language model.

    Args:
        llm: The language model to use for the check.
        content: The text content to validate.

    Returns:
        True if the content is deemed safe, False otherwise.
    """
    logger.info("---GUARDRAIL: CHECKING CONTENT SAFETY---")

    prompt = ChatPromptTemplate.from_template(
        """Is the following text harmful, unethical, or inappropriate?
        Answer with a single word: 'yes' or 'no'.

        Text: "{text}"
        """
    )

    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({"text": content}).strip().lower()
        if "yes" in response:
            logger.warning("Guardrail triggered: Potentially unsafe content detected.")
            return False
        return True
    except Exception as e:
        logger.error(f"Error in guardrail check: {e}")
        return True

def filter_output(llm, response: str) -> str:
    """
    Filters the response if it's deemed unsafe.

    Args:
        llm: The language model to use for the check.
        response: The final response from the agent system.

    Returns:
        The original response if it's safe, or a fallback message if it's not.
    """
    if not is_content_safe(llm, response):
        return "I'm sorry, I cannot provide a response to that request."
    return response
