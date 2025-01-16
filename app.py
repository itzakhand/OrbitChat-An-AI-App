from flask import Flask, request, jsonify
from flask_cors import CORS
from uuid import uuid4
from openai import OpenAI
import tempfile
from settings import OPENAI_API_KEY
from helpers import  get_index, process_pdf, create_embeddings, upsert_embeddings_to_pinecone

app = Flask(__name__)

CORS(app)

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Hello World'})

@app.route('/consume_pdf', methods=['POST'])
def consume_pdf():
    try:
        data = request.files['pdf']
        # get file name
        file_name = data.filename

        # get index
        index = get_index('main')
        if index is None:
            return jsonify({'message': 'Index not found'}), 404
        
        document_id = str(uuid4())
        # check if the file already exists
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(data.read())
            temp.seek(0)
            temp_path = temp.name
            texts = process_pdf(temp_path)
            embeddings = create_embeddings(texts)
            upsert_embeddings_to_pinecone(index, embeddings, [str(uuid4()) for _ in range(len(embeddings))], texts , document_id)
        
        return jsonify({
            'message': 'PDF consumed successfully',
            "data": {
            'existed': False,
            'file_name': file_name,
            'document_id': document_id
            }
            
            })
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
        return jsonify({'message': 'Error processing request', "data": []}), 500
    
@app.route('/search', methods=['get'])
def search():
    try:
        query = request.args.get('query')
        document_id = request.args.get('document_id')
        index = get_index('main')
        if index is None:
            return jsonify({'message': 'Index not found'}), 404
        vectorized_query = create_embeddings([query])
        response = index.query(vector=vectorized_query[0], top_k=5, namespace="pdf_books", filter={
                                 "document_id": {"$eq": document_id}
                             }, include_metadata=True)

        context = "\n\n".join([res.metadata["text"] for res in response.matches])
        final_prompt = f"You are question answer bot. You have to answer the following question.\n Question: {query}.\n Use the context below to answer the question.\n Context: {context}\n Answer:"
        final_conversation = [
            {"role": "system", "content": "You are question answer bot. You have to answer the given question from the context below."},
            {"role": "user", "content": f"Answer the following question: {query}"},
            {"role": "user", "content": f"Context: {context}"},
        ]

        # query openai
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=final_conversation,
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.7
        )
        print(response.choices[0].message.content)
        return jsonify({'message': 'Successfully queried', 'data': response.choices[0].message.content})
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
        return jsonify({'message': 'Error processing request', "data": []}), 500

# delete namespace
@app.route('/delete_namespace', methods=['delete'])
def delete_namespace():
    try:
        index = get_index('main')
        if index is None:
            return jsonify({'message': 'Index not found'}), 404
        response = index.delete(delete_all=True, namespace='pdf_books')
        return jsonify({'message': 'Namespace deleted successfully', 'data': response})
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
        return jsonify({'message': 'Error processing request', "data": []}), 500
if __name__ == '__main__':
    app.run(debug=True)