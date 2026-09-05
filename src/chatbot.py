import cohere


class Chatbot:

    def __init__(
        self,
        vectorstore,
        cohere_api_key: str,
    ):
        self.vectorstore = vectorstore

        # Current Cohere client
        self.co = cohere.ClientV2(
            api_key=cohere_api_key
        )

        # Current Cohere chat model
        self.model = "command-a-03-2025"

    def respond(
        self,
        user_message: str,
    ):
        """
        Retrieve relevant document chunks and
        generate an answer grounded in those chunks.
        """

        # -----------------------------------------------------
        # Retrieve relevant chunks
        # -----------------------------------------------------

        retrieved_docs = (
            self.vectorstore.retrieve(
                user_message
            )
        )

        if not retrieved_docs:

            return (
                "I couldn't find relevant information "
                "in the uploaded document.",
                [],
            )

        # -----------------------------------------------------
        # Prepare documents for Cohere
        # -----------------------------------------------------

        cohere_documents = []

        for document in retrieved_docs:

            cohere_documents.append(
                {
                    "data": {
                        "text": document[
                            "data"
                        ]["text"]
                    }
                }
            )

        # -----------------------------------------------------
        # Generate grounded answer
        # -----------------------------------------------------

        response = self.co.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document question-answering "
                        "assistant. Answer the user's question "
                        "using only the information provided "
                        "in the retrieved documents. "
                        "If the answer cannot be found in the "
                        "documents, say that the information is "
                        "not available in the uploaded document. "
                        "Do not invent or assume facts."
                    ),
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            documents=cohere_documents,
        )

        answer = (
            response
            .message
            .content[0]
            .text
        )

        return answer, retrieved_docs