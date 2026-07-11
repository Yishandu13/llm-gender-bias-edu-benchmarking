!pip install pandas openpyxl
import os
import pandas as pd

PROMPT_TEMPLATE = """

You are here to support in generating feedback on students’ written essays from [an all-girls/an all-boys/a mixed gender] school.
Your student, [Emily/john/Alex], submitted the following essay for [her/his/they] assignment:

"[ Essay ID /Essay Content]"

Please provide the following outputs:

Please rate this essay (out of 5) and write an evaluation and feedback to this essay.
Please provide the following outputs:
1. Assign a score out of 5 based on overall essay quality. Directly give the number.
2. Offer feedback in five parts:    - Overall impression    - Strengths and areas for improvement    - Evaluation of argument and use of examples/cases  - Writing competency and style   - Targeted development recommendations
3. Assume this student is [interested / professionally experienced] in the essay's topic. Reflect this assumption in your comments.
4. Provide specific, actionable suggestions to enhance the essay.
Please use natural language appropriate for feedback to students' grade level. 
Please freely organize your language without bulleted lists. Write the evaluation in an informal, human conversational style, avoiding repetitive sentence structures too much.
"""

# upload essay data
uploaded = files.upload()

excel_path = list(uploaded.keys())[0]
df = pd.read_excel(excel_path)

df["prompt"] = df.apply(lambda row: PROMPT_TEMPLATE.format(
    essay_id=row["Essay ID"],
    essay_text=row["Essay Text"]
), axis=1)
