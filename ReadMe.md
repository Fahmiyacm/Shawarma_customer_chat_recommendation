# Shawarma DKENZ Customer App – Summary Documentation

### Overview

The Shawarma DKENZ Customer App is a Streamlit-based web app that lets customers:

**Browse menu**

**Place orders**

**Get personalized recommendations**

**Get popular items**

**Chat with a Groq-powered LLM chatbot**

It uses Neon PostgreSQL for storage, FAISS for semantic search, and deep learning models for recommendations.

## Folder Structure

**faiss/** – Chatbot FAISS vector index

**MenuBook/** – Menu images

**Data/** – Q&A knowledge base (CSV)

**screenshots/** – App documentation images

**model_recommend.py** – Item recommendation logic

**model_evaluator.py** – Evaluates recommendation models

**reg_helper.py** – UAE phone number validation

**db_helper.py** – Neon DB operations

**build_vectorstore.py** – Builds FAISS index

**Shawarma_app.py** – Main customer-facing Streamlit app

**.env** – API keys & DB config

**requirements.txt** – Python dependencies

**README.md** – Project overview

**DETAILED_DOCUMENTATION.md** – Full documentation

## Features

**1. Menu Browsing**

Paginated menu with category filter

View via UI or image book (MenuBook/)

**2. Order Placement**

Select item, type (Normal/Spicy), and quantity

Add to cart and submit (requires valid UAE number)

**3. Chatbot**

Groq-powered (via LangChain) and limited to menu Q&A

Uses FAISS vector search with English-only responses

**4. Recommendations**

Popular items (last 4 months)

Personalized (cosine similarity or deep learning model)

**5. Cart Management**

Add/remove items

Submit order with generated order_id

## Database Schema

**menu Table:**

id, item_name, category, item_price

**orders Table:**

order_id, item_id, item_name, item_price, quantity, total_price, time_at, phone_number, type

## Setup Instructions

git clone <repo_url>

cd ShawarmaCustomerApp

pip install -r requirements.txt

Create .env:

DATABASE_URL=your_neon_url

GROQ_API_KEY=your_groq_key

Create tables and build FAISS index:

python build_vectorstore.py

**Train recommender:**

python -c "from model_recommend import model_deeplearning; model_deeplearning('<valid_UAE_number>')"

Run app:

streamlit run Shawarma_app.py

#  Usage

Start with UAE phone number (e.g., +971559745005)

**Use sidebar:**

About Us

Menu Book

Chatbot

Place Order

Popular Items

Recommendations

Cart

Example: Order “Shawarma Chicken Large” (Spicy, 2 qty) and submit.

## Troubleshooting

Chatbot wrong answers? → python build_vectorstore.py

Bad recommendations? → Retrain model

DB errors? → Check .env and run: SELECT * FROM menu LIMIT 5;

## Conclusion

The Shawarma DKENZ Customer App delivers a seamless and interactive customer experience by combining a user-friendly Streamlit interface with powerful backend technologies such as Neon PostgreSQL, FAISS semantic search, and deep learning-based recommendations. It enables efficient menu browsing, order placement, personalized recommendations, and intelligent chatbot support, all tailored to the UAE market with phone number validation.

With clear setup instructions, modular code organization, and robust features, this app serves as a solid foundation for enhancing customer engagement and boosting sales in a modern food service environment. Regular maintenance—such as rebuilding the FAISS index and retraining recommendation models—ensures continued accuracy and relevance.

Overall, Shawarma DKENZ is a practical example of integrating AI-driven tools into a real-world business application.