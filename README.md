

## ⚙️ Setup Instructions

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd Resume-Screening
```

### 2. Set up environment variables
Create a .env file and add your Groq API token:
```bash 
GROQ_API_KEY=your_api_key_here
```

Use python 3.10.14 version if running local virtual env. 

## 3. Run the App :

```bash 
deactivate
rm -rf venv
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
