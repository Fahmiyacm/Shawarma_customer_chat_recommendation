# **Shawarma DKENZ Customer App Detailed Documentation**

## **Project Overview**

The Shawarma DKENZ Customer App (Shawarma_app.py) is a Streamlit-based web application enabling customers to browse a 43-item menu (e.g., Shawarma Chicken Medium, Strawberry Mojito), place orders, view recommendations, and interact with a Grok-powered chatbot. It integrates with a Neon PostgreSQL database via db_helper.py, uses model_recommend.py for recommendations, and employs FAISS for chatbot semantic search.

##### **Folder Structure**

ShawarmaCustomerApp/

├── .github/                    # workflows folder foe creating ci cd pipeline

├── faiss/                      # FAISS index for chatbot semantic search

├── MenuBook/                   # Menu images (image_0.jpeg to image_3.jpeg)

├── Data/                       # Knowledge Base Q and A about shop

├── screenshots/                # Screenshots for documentation

├── model_recommend.py          # Cosine and deep learning recommendation models

├── model_evaluator.py          # Evaluates recommendation models

├── reg_helper.py               # UAE phone number validation

├── db_helper.py                # Neon database operations

├── build_vectorstore.py        # FAISS index creation

├── app.py             # Main Streamlit customer app

├── .env                        # Environment variables (DATABASE_URL, GROQ_API_KEY)

├── requirements.txt            # Dependencies

├── README.md                   # Short project overview

├── DETAILED_DOCUMENTATION.md   # Detailed documentation

├── .gitignore                  # Git ignore file

└── .dockerignore               # Docker ignore file

##### **Architecture**

**Frontend:** Streamlit interface with sections for chatbot , menu browsing, ordering, recommendations, and popular items.

**Backend:**

db_helper.py: Database operations (menu retrieval, order insertion, popular items,user history).

model_recommend.py: Cosine similarity (model_cosine) and neural network (model_deeplearning) recommendations.

model_evaluator.py: Selects best recommendation model.

reg_helper.py: Validates UAE phone numbers.

build_vectorstore.py: Generates FAISS index for chatbot.

LangChain with Groq (allam-2-7b) and FAISS (sentence-transformers/all-MiniLM-L6-v2).


**Database:** Neon PostgreSQL with menu and orders tables.

**Environment:** .env file for DATABASE_URL and GROQ_API_KEY.

##### **Key Features**

**Menu Browsing:**

View 44 items by category (Shawarma, Burgers, Sauces, etc.) or in MenuBook/ images.
Paginated menu book with navigation buttons.


**Order Placement:**
Select items, type (Normal/Spicy), and quantity (1–20); add to cart.
Orders saved to orders table with unique order_id.
Requires UAE phone number validation.


**Chatbot:**
Grok LLM via LangChain, constrained to 44 menu items.
FAISS vector store for semantic search (faiss/).
English-only responses, avoiding non-menu items .


**Recommendations:**
Popular items: Top 5 by quantity (last 4 months) via get_popular_items.
Personalized: model_cosine (collaborative filtering with category fallback) or model_deeplearning .


**Cart Management:**
View, remove, or submit orders.



##### Screenshots
###### Screenshots\folder includes

**Welcome Screen:** Shows welcome message, phone input, and “Start Chat” button.

**Place Order:** Shows category dropdown, item selection, type, quantity, and “Add to Cart” button.

**Chat:** Displays chat history and input field.

**Cart:** Shows cart items, “Remove” buttons, and “Submit Order” button.

**Popular Items:** Shows item dropdown and “Add Popular Item to Cart” button.

**Recommendations:** Shows personalized or popular item suggestions.

##### Database Schema:

**menu Table:**CREATE TABLE menu (
    id SERIAL PRIMARY KEY,
    item_name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    item_price NUMERIC(10, 2) NOT NULL
);


**orders Table:**CREATE TABLE orders (
    order_id VARCHAR(50),
    item_id INTEGER REFERENCES menu(id),
    item_name VARCHAR(255),
    item_price NUMERIC(10, 2),
    quantity INTEGER,
    total_price NUMERIC(10, 2),
    time_at TIMESTAMP,
    phone_number VARCHAR(20),
    type VARCHAR(50)
);

##### Dependencies

See requirements.txt in root directory.
Setup Instructions

Clone Repository:git clone <repository_url>
cd ShawarmaCustomerApp


**Install Dependencies:**pip install -r requirements.txt


Configure Environment:
Create .env:DATABASE_URL=<UR DATABASE URL>
GROQ_API_KEY=your_grok_api_key


**Database Setup**:Create tables MENU and Orders(see schema above).

**Build FAISS Index**:python build_vectorstore.py

**Menu Images**:Place image_0.jpeg to image_3.jpeg in MenuBook/.

**Train Recommendation Model**:python -c "from model_recommend import model_deeplearning; model_deeplearning('valid UAE number')"

**Run App**:streamlit run Shawarma_app.py

##### Usage

Open in browser, enter UAE phone number (e.g., +971559745005).

**Navigate:**

**About Us:** Restaurant details.

**Explore Menu Book:** View menu images.

**Chatbot:** Enquiry about Shawarma DKENZ

**Place Order:** Add items to cart.

**Popular Items:** Order top items.

**Recommendations:** View personalized suggestions.

**Cart:** Manage and submit orders.


Example: Add “Shawarma Chicken Large” (Spicy, 2), submit order.

## Troubleshooting

### Chatbot Errors:

Rebuild FAISS index if non-menu items appear:python build_vectorstore.py


### Recommendations:

Retrain model if recommender_model.pth is outdated:rm recommender_model.pth
python -c "from model_recommend import model_deeplearning; model_deeplearning('+971559284373')"

### Database:

Verify DATABASE_URL and Neon permissions:SELECT * FROM menu LIMIT 5;

Logs:
Check logging.INFO and logging.ERROR.

### Conclusion

The Shawarma DKENZ Customer App delivers a seamless and interactive customer experience by combining a user-friendly Streamlit interface with powerful backend technologies such as Neon PostgreSQL, FAISS semantic search, and deep learning-based recommendations. It enables efficient menu browsing, order placement, personalized recommendations, and intelligent chatbot support, all tailored to the UAE market with phone number validation.

With clear setup instructions, modular code organization, and robust features, this app serves as a solid foundation for enhancing customer engagement and boosting sales in a modern food service environment. Regular maintenance such as rebuilding the FAISS index and retraining recommendation models ensures continued accuracy and relevance.

Overall, Shawarma DKENZ is a practical example of integrating AI driven tools into a real world business application.


