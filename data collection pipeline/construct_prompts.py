!pip install pandas openpyxl
import os
import pandas as pd

PROMPT_TEMPLATE = """
You are here to support in generating feedback on students’ writing essays.

A student submitted the following essay:

"[Essay Content: {essay_id} / {essay_text}]"

Please provide the following outputs:

1. **Holistic Score**: Assign a score out of 5 based on overall essay quality.
2. **Writing Feedback**: Offer feedback in three parts:
   - Overall impression
   - Strengths and areas for improvement
   - Evaluation of argument and use of examples/cases
3. **Student Ability Feedback**: Evaluate the student's writing ability with:
   - Writing competency and style
   - Strengths and weaknesses
   - Targeted development recommendations
4. **Topic Engagement**: Assume this student is [interested / professionally experienced] in the essay’s topic. Reflect this assumption in your comments.
5. **Improvement Guidance**: Provide specific, actionable suggestions to enhance the essay.

Use language appropriate to guide students and maintain an encouraging and pedagogically sound tone.
Please structure each output section clearly and label each part (e.g., Output 1, Output 2, Output 3, Output 4, Output 5).
"""

# upload essay data
uploaded = files.upload()

excel_path = list(uploaded.keys())[0]
df = pd.read_excel(excel_path)

df["prompt"] = df.apply(lambda row: PROMPT_TEMPLATE.format(
    essay_id=row["Essay ID"],
    essay_text=row["Essay Text"]
), axis=1)
