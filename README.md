# Market-basket-analysis
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# Load the dataset
file_path = "/mnt/data/market_basket_dataset.csv"
df = pd.read_csv(file_path)

# Preprocessing: Group items by BillNo to create transactions
transactions = df.groupby('BillNo')['Itemname'].apply(list)

# Create a one-hot encoded DataFrame for transactions with quantities
unique_items = sorted(set(item for sublist in transactions for item in sublist))
one_hot = pd.DataFrame(0, index=transactions.index, columns=unique_items)

# Populate the one-hot matrix with quantities
for index, items in transactions.items():
    for item in items:
        one_hot.loc[index, item] += 1

# Convert the one-hot matrix to binary format (1 for presence, 0 for absence)
binary_one_hot = one_hot.copy()
binary_one_hot[binary_one_hot > 0] = 1

# Minimum support for frequent itemset mining
min_support = 0.05

# Apply Apriori algorithm
frequent_itemsets_apriori = apriori(binary_one_hot, min_support=min_support, use_colnames=True)
rules_apriori = association_rules(frequent_itemsets_apriori, metric="lift", min_threshold=1) if not frequent_itemsets_apriori.empty else pd.DataFrame()

# Apply FP-Growth algorithm
frequent_itemsets_fpgrowth = fpgrowth(binary_one_hot, min_support=min_support, use_colnames=True)
rules_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric="lift", min_threshold=1) if not frequent_itemsets_fpgrowth.empty else pd.DataFrame()

# Compare the two algorithms
comparison = {
    "Algorithm": ["Apriori", "FP-Growth"],
    "Frequent Itemsets": [len(frequent_itemsets_apriori), len(frequent_itemsets_fpgrowth)],
    "Association Rules": [len(rules_apriori), len(rules_fpgrowth)],
}
comparison_df = pd.DataFrame(comparison)

# Save the results
frequent_itemsets_apriori.to_csv("/mnt/data/frequent_itemsets_apriori.csv", index=False)
rules_apriori.to_csv("/mnt/data/rules_apriori.csv", index=False)
frequent_itemsets_fpgrowth.to_csv("/mnt/data/frequent_itemsets_fpgrowth.csv", index=False)
rules_fpgrowth.to_csv("/mnt/data/rules_fpgrowth.csv", index=False)
comparison_df.to_csv("/mnt/data/algorithm_comparison.csv", index=False)

# Output file locations
print("Apriori frequent itemsets saved to: /mnt/data/frequent_itemsets_apriori.csv")
print("Apriori rules saved to: /mnt/data/rules_apriori.csv")
print("FP-Growth frequent itemsets saved to: /mnt/data/frequent_itemsets_fpgrowth.csv")
print("FP-Growth rules saved to: /mnt/data/rules_fpgrowth.csv")
print("Comparison results saved to: /mnt/data/algorithm_comparison.csv")
