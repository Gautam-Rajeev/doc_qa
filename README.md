# doc_qa
Sample repo to parse docs and carry out qa


## Running the file : 

Follow the following steps : 

```
python3 -m venv doc_qa
source doc_qa/bin/activate 
export GEMINI_API_KEY=''
python app.py
```

This enables one to use two APIS : 

```
curl -X POST http://localhost:5000/add_docs \
  -H "Content-Type: application/json" \
  -d '{"user": "afde9394-bebd-4c5b-b794-7c3a53e5885e", "document_path": "/doc_path.pdf"}'
```

```

curl -X POST http://localhost:5000/infer \
  -H "Content-Type: application/json" \
  -d '{"user": "afde9394-bebd-4c5b-b794-7c3a53e5885e", "question": "How much revenue did they make this quarter? "}'

```


The UI can be run by using :

```
python gradio.py
```
