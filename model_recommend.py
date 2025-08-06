import db_helper
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import defaultdict
import os

# ---------- Utility Functions ----------
def clean_item_name(name):
    return name.strip().lower()

# ---------- Calculating Cosine Similarity  ----------
def cosine_similarity(vec1, vec2):
    dot = sum(i * j for i, j in zip(vec1, vec2))
    norm1 = math.sqrt(sum(i * i for i in vec1))
    norm2 = math.sqrt(sum(j * j for j in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)

# ---------- Cosine Similarity Model ----------

def model_cosine(phone):
    user_history_raw = db_helper.get_user_order_history(phone)
    if not user_history_raw:
        print("No user history found, returning popular items")
        return db_helper.get_popular_items()[:5]

    user_history = [clean_item_name(item) for item in user_history_raw]
    print(f"User history: {user_history}")

    all_orders = db_helper.get_all_orders()
    if not all_orders:
        print("No orders found in database, returning popular items")
        return db_helper.get_popular_items()[:5]

    for order in all_orders:
        order['item_name'] = clean_item_name(order['item_name'])

    item_set = {order['item_name'] for order in all_orders}
    item_list = sorted(item_set)
    item_to_idx = {item: idx for idx, item in enumerate(item_list)}
    print(f"Total unique items: {len(item_list)}")

    # Build user vectors using quantities
    user_vectors = defaultdict(lambda: [0] * len(item_list))
    for order in all_orders:
        ph = order['phone_number']
        item = order['item_name']
        if item in item_to_idx:
            user_vectors[ph][item_to_idx[item]] += order.get('quantity', 1)

    # Build target user vector using quantities
    target_orders = [order for order in all_orders if order['phone_number'] == phone]
    target_vector = [0] * len(item_list)
    for order in target_orders:
        item = order['item_name']
        if item in item_to_idx:
            target_vector[item_to_idx[item]] += order.get('quantity', 1)

    user_ordered_set = set(user_history)
    print(f"User ordered set: {user_ordered_set}")
    print(f"Target vector non-zero indices: {[item_list[idx] for idx, val in enumerate(target_vector) if val > 0]}")

    # Log item frequencies
    item_counts = defaultdict(int)
    for order in all_orders:
        item_counts[order['item_name']] += order.get('quantity', 1)
    print("Top 10 item frequencies:", sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:10])

    similarity_scores = {}
    for other_phone, vec in user_vectors.items():
        if other_phone == phone:
            continue
        sim = cosine_similarity(target_vector, vec)
        if sim > 0.2:
            similarity_scores[other_phone] = sim
            user_items = [item_list[idx] for idx, val in enumerate(vec) if val > 0]
            print(f"Similar user {other_phone} (sim={sim:.3f}) ordered: {user_items}")

    print(f"Number of similar users: {len(similarity_scores)}")

    item_scores = defaultdict(float)
    for other_phone, sim in similarity_scores.items():
        vec = user_vectors[other_phone]
        for idx, count in enumerate(vec):
            item = item_list[idx]
            if count > 0 and item not in user_ordered_set:
                print(f"Adding score for item {item} from user {other_phone}: {sim * count}")
                item_scores[item] += sim * count

    recommended_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    recommended_items = [item for item, _ in recommended_items][:5]

    # Category-based fallback
    if not recommended_items:
        print("No collaborative filtering recommendations, using category-based fallback")
        user_categories = set()
        menu = db_helper.get_all_menu_items()  # Use get_all_menu_items instead of get_menu
        for item in user_history:
            for menu_item in menu:
                if clean_item_name(menu_item['item_name']) == item:
                    user_categories.add(menu_item['category'])
        print(f"User categories: {user_categories}")

        category_counts = defaultdict(int)
        for order in all_orders:
            item = order['item_name']
            for menu_item in menu:
                if clean_item_name(menu_item['item_name']) == item and menu_item['category'] in user_categories:
                    if item not in user_ordered_set:
                        category_counts[item] += order.get('quantity', 1)

        recommended_items = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        recommended_items = [item for item, _ in recommended_items]
        print(f"Category-based recommendations: {recommended_items}")

    print("Recommended items:", recommended_items)
    print("Cosine similarity scores with other users:", similarity_scores)
    print("Item scores:", item_scores)

    return recommended_items if recommended_items else db_helper.get_popular_items()[:5]

# ---------- Deep Learning Model ----------

class RecommenderNN(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(RecommenderNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, input_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def prepare_training_data(all_orders, item_list, item_to_idx):
    user_vectors = defaultdict(lambda: torch.zeros(len(item_list)))
    for order in all_orders:
        phone = order['phone_number']
        item = clean_item_name(order['item_name'])
        if item in item_to_idx:
            user_vectors[phone][item_to_idx[item]] += order.get('quantity', 1)
    return [(vec.clone(), vec.clone()) for vec in user_vectors.values()]

def train_model(model, data, path="recommender_model.pth", epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input_vec, target_vec in data:
            optimizer.zero_grad()
            output = model(input_vec)
            loss = criterion(output, target_vec)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(data)}")
    torch.save(model.state_dict(), path)

def model_deeplearning(phone):
    history = db_helper.get_user_order_history(phone)
    all_orders = db_helper.get_all_orders()

    if not history or not all_orders:
        return db_helper.get_popular_items()[:5]

    item_set = {clean_item_name(order['item_name']) for order in all_orders}
    item_list = sorted(item_set)
    item_to_idx = {item: idx for idx, item in enumerate(item_list)}
    reverse_map = {idx: item for item, idx in item_to_idx.items()}
    input_vector = torch.zeros(len(item_list))

    for item in history:
        item = clean_item_name(item)
        if item in item_to_idx:
            input_vector[item_to_idx[item]] = 1

    model = RecommenderNN(input_size=len(item_list), hidden_size=32)
    model_path = "recommender_model.pth"
    retrain = False

    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
        except RuntimeError as e:
            print("Model architecture mismatch, retraining:", e)
            retrain = True
    else:
        retrain = True

    if retrain:
        training_data = prepare_training_data(all_orders, item_list, item_to_idx)
        if not training_data:
            return db_helper.get_popular_items()[:5]
        train_model(model, training_data, path=model_path)

    model.eval()
    with torch.no_grad():
        output = model(input_vector)

    user_ordered = set(clean_item_name(item) for item in history)
    top_indices = output.argsort(descending=True).tolist()
    recommendations = []

    for idx in top_indices:
        item = reverse_map[idx]
        if input_vector[idx] == 0 and item not in user_ordered:
            recommendations.append(item)
        if len(recommendations) == 5:
            break

    return recommendations if recommendations else db_helper.get_popular_items()[:5]

# ---------- Test ----------

if __name__ == "__main__":
    phone = "+971559284373"
    print("Order History:", db_helper.get_user_order_history(phone))
    print("Cosine Recommendations:", model_cosine(phone))
    print("Deep Learning Recommendations:", model_deeplearning(phone))
    print("Popular Items:", db_helper.get_popular_items()[:5])


