import pandas as pd
import re

df = pd.read_excel("your data")
df.columns = ['ID', 'essay', 'original_word', 'counterfactual_word'] #you can replace with your data structure

word_map = dict(zip(df['original_word'].str.lower(), df['counterfactual_word']))
unique_essays = df['essay'].dropna().astype(str).unique()

def match_case(original, replacement):
    if original.isupper():
        return replacement.upper()
    elif original[0].isupper():
        return replacement.capitalize()
    else:
        return replacement.lower()

def replace_with_highlight(text, word_map):
    def replacer(match):
        original = match.group(0)
        replacement = word_map.get(original.lower(), original)
        replaced = match_case(original, replacement)
        return f"<span style='background-color:yellow;font-weight:bold;'>{replaced}</span>"

    pattern = r'\b(' + '|'.join(re.escape(w) for w in word_map) + r')\b'
    return re.sub(pattern, replacer, text, flags=re.IGNORECASE)

def replace_plain(text, word_map):
    def replacer(match):
        original = match.group(0)
        replacement = word_map.get(original.lower(), original)
        return match_case(original, replacement)

    pattern = r'\b(' + '|'.join(re.escape(w) for w in word_map) + r')\b'
    return re.sub(pattern, replacer, text, flags=re.IGNORECASE)


results = []
html_table_rows = ""

for text in unique_essays:
    highlighted = replace_with_highlight(text, word_map)
    plain = replace_plain(text, word_map)

    results.append({
        "original_text": text,
        "counterfactual_plain": plain
    })

    html_table_rows += f"<tr><td>{text}</td><td>{highlighted}</td></tr>\n"

df_plain = pd.DataFrame(results)
df_plain.to_excel("counterfactualessay_plain.xlsx", index=False)

'''
You could also construct an HTML file for better visualisation by following codes

html_str = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Counterfactual Highlight</title>
<style>
    body {{ font-family: Arial, sans-serif; padding: 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 10px; vertical-align: top; }}
    th {{ background-color: #f2f2f2; }}
</style>
</head>
<body>
<h2>Counterfactual Text Highlighting</h2>
<table>
<tr><th>Original Essay</th><th>Counterfactual with Highlight</th></tr>
{html_table_rows}
</table>
</body>
</html>
"""

with open("malecounterfactualessay_highlighted.html", "w", encoding="utf-8") as f:
    f.write(html_str)

'''
