import db_helper
import model_recommend
from collections import defaultdict

def evaluate_models(phone):
    # Ground truth = actual items user ordered
    actual_items = set(db_helper.get_user_order_history(phone))
    # Get recommendations from both models
    cosine_items = set(model_recommend.model_cosine(phone))
    deep_items = set(model_recommend.model_deeplearning(phone))

    # Get user categories from order history
    menu = db_helper.get_all_menu_items()
    user_categories = set()
    for item in actual_items:
        for menu_item in menu:
            if menu_item['item_name'].lower() == item.lower():
                user_categories.add(menu_item['category'])

    def category_precision(recommended, categories):
        if not recommended:
            return 0.0
        relevant = 0
        for item in recommended:
            for menu_item in menu:
                if menu_item['item_name'].lower() == item.lower() and menu_item['category'] in categories:
                    relevant += 1
                    break
        return relevant / len(recommended)

    # Calculate category precision score
    precision_cosine = category_precision(cosine_items, user_categories)
    precision_deep = category_precision(deep_items, user_categories)

    print(f"User categories: {user_categories}")
    print(f"Cosine recommendations: {cosine_items}")
    print(f"Deep learning recommendations: {deep_items}")
    print(f"Category precision (cosine): {precision_cosine:.3f}")
    print(f"Category precision (deep): {precision_deep:.3f}")

    # Return best model based on category precision
    return "cosine" if precision_cosine >= precision_deep else "deep"

if __name__ == "__main__":
    print(evaluate_models("+971555249922"))



