from quart import Quart, request, jsonify
from model import Inference

# Create the Quart app instance
app = Quart(__name__)

# Instantiate our asynchronous Inference class
inference = Inference()

# POST endpoint to add a document for a given user id
@app.route('/add_docs', methods=['POST'])
async def add_docs():
    data = await request.get_json()
    user = data.get("user")
    document_path = data.get("document_path")
    if not user or not document_path:
        return jsonify({"error": "Missing user or document_path"}), 400
    try:
        # Call the asynchronous add_document method
        result = await inference.add_document(user, document_path)
        return jsonify({"message": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# POST endpoint to perform inference (ask a question) for a given user id
@app.route('/infer', methods=['POST'])
async def infer():
    data = await request.get_json()
    user = data.get("user")
    question = data.get("question")
    if not user or not question:
        return jsonify({"error": "Missing user or question"}), 400
    try:
        # Call the asynchronous infer method
        answer = await inference.infer(user, question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()
