"""Shawarma DKENZ Streamlit Application
This application provides an interactive interface for customers to browse the menu,
place orders, view recommendations, and interact with a chatbot for a shawarma restaurant"""


# --- Standard Libraries ---
import os
import logging
from datetime import datetime

# --- Third-party Libraries ---
from dotenv import load_dotenv  # For loading environment variables from .env
import streamlit as st  # Web UI
from PIL import Image  # Image processing

# --- Custom Modules ---
import model_recommend        # Module to generate product recommendations
import mode_evaluator         # Module to evaluate recommendation performance

# --- LangChain Community Modules ---
from langchain_community.vectorstores import FAISS  # For semantic search storage
from langchain_community.embeddings import HuggingFaceEmbeddings  # To create sentence embeddings

# --- LangChain Models & Chains ---
from langchain_groq import ChatGroq  # Integration with Groq LLM for chatbot
from langchain.prompts import PromptTemplate  # Template to format prompts for LLMs
from langchain.chains import LLMChain  # Used to build conversational chain

# Custom modules
from db_helper import (
    get_categories,
    get_items_by_category,
    get_item_price,
    exact_match_menu_item,
    insert_order,
    get_next_order_id,
    get_popular_items,
    get_all_menu_items,
    get_user_order_history,
)
from reg_helper import format_uae_number, is_valid_uae_number

# Streamlit page configuration (must be first Streamlit command)
st.set_page_config(page_title="Shawarma DKENZ", layout="wide")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Custom CSS for styling
st.markdown(
    """
    <style>
    img {
        pointer-events: none; /* Disables all clicks on images */
    }
    .custom-button {
        background-color: #f63366;
        color: white;
        border: none;
        padding: 0.5em 1em;
        border-radius: 12px;
        cursor: pointer;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def initialize_session_state():
    """Initialize session state variables if not already set."""
    session_keys = {
        "messages": [],
        "cart": [],
        "show_menu": False,
        "show_popular": False,
        "show_recommend": False,
        "show_cart": False,
        "show_menu_book": False,
        "show_chat": False,
        "menu_book_page": 0,
        "phone_entered": False,
        "chat_history": [],
        "show_about": False,
    }
    for key, default_value in session_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize session state
initialize_session_state()

# Menu book pages
MENU_PAGES = [
    "Menubook/image_0.jpeg",
    "Menubook/image_1.jpeg",
    "Menubook/image_2.jpeg",
    "Menubook/image_3.jpeg",
]

def display_welcome_message():
    """Display welcome message and phone input for unauthenticated users."""
    st.markdown(
        """
        <div style="text-align: center; font-size: 20px;">
         <strong><h3>Welcome to Shawarma DKENZ!... Chat Bot</h3></strong>
        📞 Order now: +971 559745005 / 0526746479<br>
        📍 Location: Hamidiya 1, Ajman, UAE<br>
        🗞️ Get our delicious shawarmas, family meals, and more!<br><br><br><br><br>
        </div>
        """,
        unsafe_allow_html=True
    )
    col_left, col_center, col_right = st.columns([1, 3, 1])
    with col_center:
        phone_input = st.text_input(
            "Enter Your Valid UAE Phone Number to Start (e.g., +971XXXXXXXXX)"
        )
    col_left_btn, col_center_btn, col_right_btn = st.columns([3, 1, 3])
    with col_center_btn:
        if st.button("Start Chat"):
            if phone_input.strip():
                if is_valid_uae_number(phone_input):
                    st.session_state.phone_entered = True
                    st.session_state.user_phone = format_uae_number(phone_input)
                    st.success("Welcome! You can now start ordering.")
                    logger.info(f"User authenticated with phone: {st.session_state.user_phone}")
                    st.rerun()
                else:
                    st.error("❌ Please enter a valid UAE phone number.")
                    logger.warning("Invalid phone number entered")
    st.stop()

def reset_visibility_flags():
    """Reset all visibility flags to False."""
    visibility_flags = [
        "show_about",
        "show_menu",
        "show_popular",
        "show_recommend",
        "show_cart",
        "show_menu_book",
        "show_chat",
    ]
    for flag in visibility_flags:
        st.session_state[flag] = False

# Phone validation
if not st.session_state.phone_entered:
    display_welcome_message()

# Sidebar navigation
st.sidebar.markdown("### 🧭 **Choose Your Destination**")
sidebar_selection = st.sidebar.radio(
    "",
    [
        "📖 About Us",
        "🧾 Explore Menu Book",
        "💬 Chat with our BOT",
        "📝 Place Your Order",
        "🔥 Popular Items",
        "🌟 Recommendations",
        "🛒 Your Cart",
    ]
)

# Reset visibility flags and set based on sidebar selection
reset_visibility_flags()
if sidebar_selection == "📖 About Us":
    st.session_state.show_about = True
elif sidebar_selection == "🧾 Explore Menu Book":
    st.session_state.show_menu_book = True
elif sidebar_selection == "📝 Place Your Order":
    st.session_state.show_menu = True
elif sidebar_selection == "🔥 Popular Items":
    st.session_state.show_popular = True
elif sidebar_selection == "🌟 Recommendations":
    st.session_state.show_recommend = True
elif sidebar_selection == "🛒 Your Cart":
    st.session_state.show_cart = True
elif sidebar_selection == "💬 Chat with our BOT":
    st.session_state.show_chat = True

# About Us section
if st.session_state.show_about:
    st.markdown("<h2 style='text-align: center;'>About Shawarma DKENZ</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        Shawarma DKENZ is a brand born out of passion and flavor, beginning its journey seven years ago in Saudi Arabia and expanding to the UAE in 2024. It’s not just a restaurant — it’s a story of dedication, partnership, and a deep love for the art of charcoal-grilled shawarma. Run by a group of committed partners instead of a single CEO, the team at DKENZ works together to deliver the rich, smoky flavors of Turkish-style shawarma using their own blend of spices and traditional methods. The menu revolves around this signature charcoal shawarma, enhanced by a variety of refreshing juices, hot and cold beverages, and flavorful sides, all crafted to leave a lasting impression. At Shawarma DKENZ, every bite carries a part of our story, and we welcome you to be part of it. For any inquiries or to connect with us, feel free to call 📞 +971 559745005 or +971 526746479.
        """
    )
    logger.info("Displayed About Us section")

# Chatbot section
if st.session_state.show_chat:
    st.markdown(
        """
        <div style="text-align: center; font-size: 20px;">
         <strong><h3>Welcome to Shawarma DKENZ!... Chat Bot</h3></strong>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display chat history
    for message in st.session_state.chat_history:
        role, content = message
        with st.chat_message("user" if role == "You" else "assistant"):
            st.markdown(content)

    # Chat input
    user_query = st.chat_input("Type your message...")
    if user_query:
        try:
            # Save user message
            st.session_state.chat_history.append(("You", user_query))
            with st.chat_message("user"):
                st.markdown(user_query)
                logger.info(f"User query: {user_query}")

            # Load embeddings and vector store
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.load_local(
                "faiss_index",
                embeddings,
                allow_dangerous_deserialization=True
            )
            retrieved_docs = vectorstore.similarity_search(user_query, k=5)

            # Fallback if no document found
            if not retrieved_docs or all(len(doc.page_content.strip()) == 0 for doc in retrieved_docs):
                response = "Sorry, I can only help with our menu. Please call +971-559745005 for more details."
                logger.warning("No relevant documents found or docs are empty")
            else:
                answers_text = "\n\n".join([f"- {doc.page_content.strip()}" for doc in retrieved_docs])
                prompt_template = """
                You are a friendly team member at Shawarma DKENZ. Only use the menu items listed in the knowledge base below to answer the customer.

                Customer question: "{user_question}"

                Knowledge base:
                {answers}

                Important Rules:
                - Only answer based on the knowledge base above.
                - Do not mention desserts, beef shawarma, falafel, hummus, cookies, or anything not in the knowledge base.
                - If the answer is not in the knowledge base, say: "Sorry, I can only help with our menu. Please call +971-559745005 for more details."
                - Use a friendly tone and respond in 1–4 short sentences.
                - If prices are asked, use exact prices from the knowledge base.
                - Do not mention you are an AI or assistant.
                - Keep it warm, natural, and to the point.

                Now reply to the customer:
                """
                prompt = PromptTemplate(
                    input_variables=["user_question", "answers"],
                    template=prompt_template.strip()
                )
                llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="allam-2-7b")
                chain = LLMChain(llm=llm, prompt=prompt)
                response = chain.run(user_question=user_query, answers=answers_text)
                logger.info("Generated chatbot response successfully")

        except Exception as e:
            response = "Sorry, something went wrong. Please try again or call us at +971-555-123456."
            logger.error(f"Error generating chatbot response: {e}")

        # Save and display bot response
        st.session_state.chat_history.append(("Shawarma Bot", response))
        with st.chat_message("assistant"):
            st.markdown(response)

# Menu Book section
if st.session_state.show_menu_book:
    col_left, col_center, col_right = st.columns([1, 7, 1])
    with col_left:
        if st.button("⬅️") and st.session_state.menu_book_page > 0:
            st.session_state.menu_book_page -= 1
            logger.info("Navigated to previous menu page")
    with col_right:
        if st.button("➡️") and st.session_state.menu_book_page < len(MENU_PAGES) - 1:
            st.session_state.menu_book_page += 1
            logger.info("Navigated to next menu page")
    try:
        img = Image.open(MENU_PAGES[st.session_state.menu_book_page])
        st.image(img, width=750)
        logger.info(f"Displayed menu page: {MENU_PAGES[st.session_state.menu_book_page]}")
    except FileNotFoundError as e:
        st.error("Menu image not found. Please contact support.")
        logger.error(f"Menu image not found: {e}")

# Place Order section
if st.session_state.show_menu:
    with st.expander("🖔️ Select an Item to Order", expanded=True):
        try:
            categories = get_categories()
            selected_category = st.selectbox("Category", categories)
            items = get_items_by_category(selected_category)

            item_labels = [f"{item['item_name']} - AED {item['item_price']:.2f}" for item in items]
            item_lookup = {label: item for label, item in zip(item_labels, items)}

            selected_label = st.selectbox("Item", item_labels)
            selected_item = item_lookup[selected_label]
            type_choice = st.radio("Choose Type", ["Normal", "Spicy"], horizontal=True)
            quantity = st.selectbox("Quantity", list(range(1, 21)))

            if st.button("✅ Add to Cart"):
                total = selected_item["item_price"] * quantity
                st.session_state.cart.append({
                    "item": selected_item["item_name"],
                    "type": type_choice,
                    "quantity": quantity,
                    "price": selected_item["item_price"],
                    "total": total,
                })
                st.success(
                    f"🛒 Added {quantity} x {selected_item['item_name']} ({type_choice}) — AED {total:.2f}"
                )
                logger.info(f"Added to cart: {quantity} x {selected_item['item_name']}")
                st.rerun()
        except Exception as e:
            st.error("Failed to load menu items. Please try again.")
            logger.error(f"Error in Place Order section: {e}")

# Cart section
if st.session_state.show_cart:
    st.markdown("### 🛒 Your Cart")
    if st.session_state.cart:
        for i, item in enumerate(st.session_state.cart):
            cols = st.columns([6, 1])
            with cols[0]:
                st.markdown(
                    f"{i+1}. {item['quantity']} x {item['item']} ({item['type']}) - AED {item['total']:.2f}"
                )
            with cols[1]:
                if st.button("Remove", key=f"remove_{i}"):
                    st.session_state.cart.pop(i)
                    logger.info(f"Removed item {i+1} from cart")
                    st.rerun()

        if st.button("📎 Submit Order"):
            try:
                order_id = get_next_order_id()
                for item in st.session_state.cart:
                    item_info = exact_match_menu_item(item["item"])
                    if item_info:
                        insert_order({
                            "order_id": order_id,
                            "item_id": item_info["id"],
                            "item_name": item["item"],
                            "item_price": item["price"],
                            "quantity": item["quantity"],
                            "total_price": item["total"],
                            "time_at": datetime.now(),
                            "phone_number": st.session_state.user_phone,
                            "type": item["type"],
                        })
                st.success("🎉 Order submitted successfully!")
                logger.info(f"Order {order_id} submitted for phone: {st.session_state.user_phone}")
                st.session_state.cart.clear()
            except Exception as e:
                st.error("Failed to submit order. Please try again.")
                logger.error(f"Error submitting order: {e}")
    else:
        st.info("🛒 Your cart is empty.")
        logger.info("Displayed empty cart message")

# Popular Items section
if st.session_state.show_popular:
    st.markdown("### 🔥 Popular Items")
    try:
        popular_raw = get_popular_items()[:20]
        if popular_raw:
            menu_items = get_all_menu_items()
            popular_items = [item for item in menu_items if item["item_name"] in popular_raw]
            item_labels = [f"{item['item_name']} - AED {item['item_price']:.2f}" for item in popular_items]
            item_lookup = {label: item for label, item in zip(item_labels, popular_items)}

            with st.expander("🖔️ Select a Popular Item to Order", expanded=True):
                selected_label = st.selectbox("Popular Items", item_labels)
                selected_item = item_lookup[selected_label]
                type_choice = st.radio("Choose Type", ["Normal", "Spicy"], horizontal=True)
                quantity = st.selectbox("Quantity", list(range(1, 21)))

                if st.button("✅ Add Popular Item to Cart"):
                    total = selected_item["item_price"] * quantity
                    st.session_state.cart.append({
                        "item": selected_item["item_name"],
                        "type": type_choice,
                        "quantity": quantity,
                        "price": selected_item["item_price"],
                        "total": total,
                    })
                    st.success(
                        f"🛒 Added {quantity} x {selected_item['item_name']} ({type_choice}) — AED {total:.2f}"
                    )
                    logger.info(f"Added popular item to cart: {quantity} x {selected_item['item_name']}")
                    st.rerun()
        else:
            st.info("No popular items found.")
            logger.info("No popular items available")
    except Exception as e:
        st.error("Failed to load popular items. Please try again.")
        logger.error(f"Error in Popular Items section: {e}")

# Recommendations section
if st.session_state.show_recommend:
    st.markdown("### 🌟 Recommendations")
    try:
        user_phone = st.session_state.user_phone
        user_history = get_user_order_history(user_phone)

        if not user_history:
            st.info("No order history found. Showing popular items instead.")
            recommended_raw = get_popular_items()[:20]
            logger.info("No user history; defaulting to popular items")
        else:
            best_model = mode_evaluator.evaluate_models(user_phone)
            if best_model == "cosine":
                recommended_raw = model_recommend.model_cosine(user_phone)
                logger.info("Using cosine model for recommendations")
            else:
                recommended_raw = model_recommend.model_deeplearning(user_phone)
                logger.info("Using deep learning model for recommendations")

        if recommended_raw:
            menu_items = get_all_menu_items()
            recommended_raw_lower = [r.lower() for r in recommended_raw]
            recommended_items = [
                item for item in menu_items
                if item["item_name"].lower() in recommended_raw_lower
            ]

            if recommended_items:
                item_labels = [
                    f"{item['item_name']} - AED {item['item_price']:.2f}" for item in recommended_items
                ]
                item_lookup = {label: item for label, item in zip(item_labels, recommended_items)}

                with st.expander("🖔️ Select a Recommended Item to Order", expanded=True):
                    selected_label = st.selectbox("Recommended Items", item_labels)
                    selected_item = item_lookup.get(selected_label)

                    if selected_item:
                        type_choice = st.radio("Choose Type", ["Normal", "Spicy"], horizontal=True)
                        quantity = st.selectbox("Quantity", list(range(1, 21)))

                        if st.button("✅ Add Recommended Item to Cart"):
                            total = selected_item["item_price"] * quantity
                            st.session_state.cart.append({
                                "item": selected_item["item_name"],
                                "type": type_choice,
                                "quantity": quantity,
                                "price": selected_item["item_price"],
                                "total": total,
                            })
                            st.success(
                                f"🛒 Added {quantity} x {selected_item['item_name']} ({type_choice}) — AED {total:.2f}"
                            )
                            logger.info(f"Added recommended item to cart: {quantity} x {selected_item['item_name']}")
                    else:
                        st.warning("⚠️ Could not find selected item details.")
                        logger.warning("Selected recommended item not found")
            else:
                st.info("😔 No matching recommended items found in the menu.")
                logger.info("No matching recommended items found")
        else:
            st.info("😔 No recommendations found.")
            logger.info("No recommendations available")
    except Exception as e:
        st.error("Failed to load recommendations. Please try again.")
        logger.error(f"Error in Recommendations section: {e}")
