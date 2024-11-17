# Backend Flask Code
- Hosted with Vercel 

# Files
- /api/index.py contains the backend code
- /api/rag.py contains the code for the RAG model
- /requirements.txt contains the required packages

# API Endpoints
## Host: https://hackutd-backend.vercel.app/
- POST: /postMessage
- Body: {"message": "Content", "convoId": "Optional"}
- Response: {"reply": "Reply", "suggestions": [{}, {}, {}]}
- When this endpoint is requested for the first time, a convoId is returned, which should be sent in the subsequent requests to maintain the context of the conversation.
- The suggestions are the suggested products, the exact info will be updated soon.