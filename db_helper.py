import os
import logging
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import Error as PsycopgError
from dotenv import load_dotenv

# --- Logging Setup ---
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()

# --- Database Connection ---
def get_connection() -> psycopg2.extensions.connection:
    """Establish and return a new database connection to Neon."""
    try:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL not found in environment variables")
        connection = psycopg2.connect(database_url)
        logger.info("Successfully connected to the Neon database")
        return connection
    except (PsycopgError, ValueError) as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

# --- Menu Functions ---
def get_item_price(item_name: str) -> float:
    """Retrieve price of a menu item by name."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT item_price FROM menu WHERE LOWER(item_name) = LOWER(%s)", (item_name,))
        result = cursor.fetchone()
        if result:
            return float(result[0])
        raise ValueError(f"Item '{item_name}' not found in menu.")
    except Exception as e:
        logger.error(f"Error in get_item_price: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def get_all_menu_items() -> List[Dict[str, Any]]:
    """Fetch all menu items as dictionaries."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute("SELECT id, item_name, category, item_price FROM menu ORDER BY id;")
        items = cur.fetchall()
        logger.info("Successfully fetched all menu items")
        return items
    except Exception as e:
        logger.error(f"Error in get_all_menu_items: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()

def get_menu_items() -> List[str]:
    """Fetch all item names from the menu table."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("SELECT item_name FROM menu")
        return [r['item_name'] for r in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Error in get_menu_items: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def get_categories() -> List[str]:
    """Fetch distinct menu categories."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT category FROM menu WHERE category IS NOT NULL ORDER BY category")
        return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Error in get_categories: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def get_items_by_category(category: str) -> List[Dict[str, Any]]:
    """Get menu items filtered by category."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("SELECT id, item_name, item_price FROM menu WHERE category = %s ORDER BY item_name", (category,))
        return cursor.fetchall()
    except Exception as e:
        logger.error(f"Error in get_items_by_category: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def exact_match_menu_item(user_input: str) -> Dict[str, Any]:
    """Find an exact match for a menu item name."""
    items = get_all_menu_items()
    return next((item for item in items if item['item_name'].lower() == user_input.lower()), None)

# --- Order Functions ---
def insert_order(order_data: Dict[str, Any]) -> None:
    """
    Insert a new order into the database.
    Expected keys in order_data: order_id, item_id, item_name, item_price,
    quantity, total_price, time_at, phone_number, type
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO orders (
                order_id, item_id, item_name, item_price, quantity,
                total_price, time_at, phone_number, type
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            order_data["order_id"], order_data["item_id"], order_data["item_name"],
            order_data["item_price"], order_data["quantity"], order_data["total_price"],
            order_data["time_at"], order_data["phone_number"], order_data["type"]
        ))
        conn.commit()
        logger.info(f"Successfully inserted order ID: {order_data['order_id']}")
    except Exception as e:
        logger.error(f"Error inserting order: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()

def get_user_order_history(phone_number: str) -> List[str]:
    """Retrieve all items ordered by a specific phone number."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT item_name FROM orders
            WHERE phone_number = %s
            ORDER BY time_at DESC
        """, (phone_number,))
        return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Error in get_user_order_history: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def get_all_orders() -> List[Dict[str, Any]]:
    """Retrieve all orders from the database."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("SELECT phone_number, item_name FROM orders")
        return cursor.fetchall()
    except Exception as e:
        logger.error(f"Error in get_all_orders: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def get_popular_items() -> List[str]:
    """Return top 5 most ordered items in the past 4 months."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("""
            SELECT TRIM(item_name) AS item_name, SUM(quantity) AS total_quantity
            FROM orders
            WHERE time_at >= NOW() - INTERVAL '4 months'
            GROUP BY TRIM(item_name)
            ORDER BY total_quantity DESC
            LIMIT 5
        """)
        return [r['item_name'] for r in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Error in get_popular_items: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def get_next_order_id() -> str:
    """Generate the next order ID based on existing orders."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT MAX(order_id) FROM orders")
        max_id = cursor.fetchone()[0]
        return str(int(max_id) + 1) if max_id else "1"
    except Exception as e:
        logger.error(f"Error in get_next_order_id: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

