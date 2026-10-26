#%%
import pandas as pd
import numpy as np
import os, sys, re
from run_ds import run_ds

#test deepseek
data_dir = 'data'
annot_dir = 'annot'
output_dir = 'output'
mode = 'combined'
use_dataset = 'final' #'train'
exp_id = '0307'
max_token = 2048

if mode == 'narrative':
    text_col = 'NARRATIVE'
elif mode == 'impression':
    text_col = 'NORMAL'
elif mode == 'clinical_corr':
    text_col = 'CLINICAL CORRELATION'
elif mode == 'combined':
    text_col = 'combined_text'

if use_dataset == 'train':
    narrat_name = f'{data_dir}/{annot_dir}/combined_cleaned.csv'
    narrat_name2 = f'{data_dir}/{annot_dir}/combined_extra66_cleaned.csv'
    output_file_name = f'combined_{mode}'
    df = pd.read_csv(narrat_name)
    df2 = pd.read_csv(narrat_name2)
    df = pd.concat([df, df2])
elif use_dataset == 'final':
    narrat_name = f'{data_dir}/notes_final.csv'
    output_file_name = f'final_{mode}'
    df = pd.read_csv(narrat_name)

output_csv_path = os.path.join(data_dir,output_dir,output_file_name)
#%%
def code_from_ans(result, full_output):
    search_text = result.lower()
    matches = re.findall(r"(yes|no|unclear)", search_text)
    if matches:
        match = matches[-1]
        if match == 'yes':
            return 1
        elif match == 'no':
            return 0
        elif match == 'unclear':
            return 2
    else: 
        return np.nan

classes = [0,1,2,3,4,5,6,9]
message_dict = {
0:f'''You are a neurologist reviewing EEG reports to determine whether the following report describes any **seizure** or **clinical event**.
Definitions:
Seizures must be explicitly described using the word “seizure”. Do **not** classify other EEG patterns (e.g., spikes, BIRDS, slowing, or ictal-interictal continuum) as seizures.
Clinical events are **abnormal** patient behavior, movements, or symptoms including but are not limited to limb jerking, shaking, twitching, moaning, dizziness, unresponsiveness, or speech problems.  
If the patient is under some abnormal state continuously, do not consider that state as an clinical event.
Reasoning Instructions:
- If the report described any seizure or clinical event, respond with Yes. 
- If the report clearly states that the patient had no seizure and no clinical event, respond with No. 
- If the report mentioned: "The following events...", with some actual description about the events, the classification is likely Yes.  
- If the report mentioned: "no clinical event...", but described some actual clinical events, respond with Yes. 
- If the report mentioned any non-accidental push button events, respond with Yes.
- Mentioning of "typical episodes were recorded" should be considered as clinical events, and you should respond with Yes.
- If the report did not mention seizure or clinical event, respond with No.
Below is the EEG Report: \n''',
1:f'''You are a neurologist reviewing an EEG report to determine if it explicitly describes an **electroclinical seizure**.
Definitions:
- An **electroclinical seizure** requires either satisfying **both** criteria below:
1. Explicit statement that a seizure event was present for this patient. The word **“seizure”(s)** must be mentioned, do **not** classify other EEG patterns (e.g., spikes, BIRDS, slowing, ictal-interictal continuum, myclonic jerks, or status epilepticus) as seizures. Ignore any content about last or previous seizure.
2. Description of any non-seizure **clinical events** that are **confirmed to happen during the seizure**. 
    Clinical events are involuntary clinical changes such as altered mental status, fall, arrhythmia, syncope, injury, shaking, twitching, unresponsiveness, speech problems, etc. 
    Mentioning of "typical episodes were recorded" should be considered as clinical events
    EEG activity or events, including seizures, are not clinical events.
    The nurses can do neurological exams during seizures, including testing of nodding, speaking, answering questions, do not consider these as clinical events.
- if the report explicitly mentioned there are "electroclinical seizure" or "clinical seizure" present, ignore the requirements above, and respond with **Yes**
Reasoning Instructions:
- If the report described any electroclinical seizure satisfying the Definitions, respond with **Yes**. 
- Otherwise, respond with No.
- If there're any ambiguity or language that casts doubt on ictal correlation (e.g. "not clearly ictal"), classify as **No**
Below is the EEG Report:\n''',
2:f'''You are a neurologist reviewing EEG reports.
Your task is to determine whether the following report describes a **subclinical seizure**.
Definitions:
- An **subclinical seizure** requires 
1. Explicit statement that a seizure event was present for this patient. The word **“seizure”(s)** must be mentioned, do **not** classify other EEG patterns (e.g., spikes, BIRDS, slowing, ictal-interictal continuum, or status epilepticus) as seizures. Ignore any content about last or previous seizure.
2. Explicit statements that **no clinical event** happened during the seizure or no **clinical event** is linked to the seizure event, typically described with phrases like "without clinical correlate", "no clinical signs".
    Clinical events are involuntary patient behavior, movements, or symptoms including but are not limited to limb movements, shaking, twitching, dizziness, unresponsiveness, or speech problems. 
    EEG activity or events, including seizure, are not clinical events.
- If it directly mentioned there are "subclinical seizure"(s), ignore the requirement above, and respond with **Yes**
    
Important Rules:
- Avoid mixing "subclinical seizure" with "electroclinical seizure" or "electrographic seizure".
**Subclinical seizure** is a seizure with an **explicit** statement that **no clinical signs** were observed. 
    1. **Electroclinical seizure**: seizure with clinical symptoms described (e.g., shaking, unresponsiveness).
    If there are any clinical events, like involuntary twitching, moaning, speech issues, associated with the seizure event, then it is an electroclinical seizure,  **NOT** an subclinical seizure
    2. **Electrographic seizure**: seizure with no mention of clinical signs (neither positive nor negative). It is completely silent on clinical correlation.
You should focus on whether the report describes any **subclinical seizure** but not the other two types.
Especially don't mix electrographic seizures (without no positive nor negative statements about clincial events) with subclinical seizures (with clear statements of no clinical events)
- if a seizure event is described as "without clear clinical correlate/events...", it should be interpret as "no clinical event", and this event is a subclinical seizure event
- The statement about clinical events or clinical correlate may be far from the seizure description. Please read the full report before giving final answer.
Reasoning Instruction:
- If the report described any **subclinical seizure** satisfying the definitions, respond with **Yes**. Otherwise, respond with **No**.
Below is the EEG Report: \n''',
3:f'''You are a neurologist reviewing an EEG report to determine whether it describes an **electrographic seizure** 
Definitions: 
- An **electrographic seizure** requires **both**:
1. Explicit statement that a seizure event was present for this patient. The word **“seizure”(s)** must be mentioned, do **not** classify other EEG patterns (e.g., spikes, BIRDS, slowing, ictal-interictal continuum, or status epilepticus) as seizures. Ignore any content about last or previous seizure.
2. The report contain **no information** about whether there are any **clinical events** present or absent.
    Clinical events are involuntary patient behavior, movements, or symptoms including but are not limited to limb movements, shaking, twitching, dizziness, unresponsiveness, or speech problems. 
    EEG activity or events, including seizure, are not clinical events.
- Or explicit use of the phrase "electrographic seizure" if not satisfying the two rules
Important Rules:
- Avoid mixing "electrographic seizure" with "electroclinical seizure" or "subclinical seizure".
**Electrographic seizure**: seizure with no mention of clinical signs (neither positive nor negative). It is completely silent on clinical correlation.
    1. **Subclinical seizure** is a seizure with an **explicit** statement that **no clinical signs** were observed. 
    If a seizure has a clinical statement like:
    - “no/without clinical correlate”
    - “no/without clinical signs”
    - “not clearly an ictal phenomenon”
    - “no clear clinical manifestation”
    - "without clear clinical correlate"
    → Then it is a subclinical seizure and is **NOT** an electrographic seizure
    2. **Electroclinical seizure**: seizure with clinical symptoms described (e.g., shaking, unresponsiveness).
    If there are any clinical events, like twitching, moaning, speech issues, associated with the seizure event, then it is an electroclinical seizure,  **NOT** an electrographic seizure
Reasoning instructions:
If the the report described any **electrographic seizure** satisfying the definitions, respond with **Yes**. Otherwise, respond with **No**.
Steps:
1. Are seizures explicitly described? (e.g., “seizure”, “ictal event”) → If Yes then continue, else Answer **No**  
2. Is there any mention of clinical events during the seizures? → If Yes then Answer **No**  
3. Is there any **explicit denial** of clinical events (e.g., “no clinical correlate”, "without clear clinical correlate", “not clearly ictal”)? → If Yes then Answer **No**  
4. Is there **no mention** at all of clinical signs? → If Yes then Answer **Yes**
Below is the EEG Report: \n''',
4:f'''You are a neurologist reviewing EEG reports.
Your task is to determine whether a non-seizure **clinical event** with **no EEG correlate** is explicitly described in the report.

A clinical event is any clinical change such as altered mental status, confusion, fall, arrhythmia, syncope, injury, etc.
EEG activity, EEG events, including seizures and status epilepticus, should not be considered as clinical events.
Any non-accidental push button events should be considered as clinical events.
Mentioning of "typical episodes were recorded" should be considered as clinical events.
Absence of clincial events, should not be considered as clinical events. 
If the patient is under some abnormal state continuously, do not consider that state as an clinical event.

A non-seizure clinical event with no EEG correlate requires BOTH:
1. A clinical event is described
2. The report includes a clear and explicit statement that there is NO EEG correlate or NO EEG change during the clinical event, using phrases such as: “No EEG correlate”, “Without EEG correlate”, “No EEG changes”, “No electrographic correlate”, **no corresponding EEG change** or other direct statement that links the event to a lack of EEG findings

Reasoning instructions:
If the the report described any **clinical event** with **no EEG correlate** satisfying the definitions, respond with **Yes**. Otherwise, respond with **No**.

Additional Constraints:
- If the report describes EEG abnormalities or findings during the clinical event, Answer **No**
- If the report does not mention EEG findings during the event, Answer **No**.
- If the report described the event as **non-ictal event**, **non-epileptic event**. Answer **Yes** 
- If the event is a "myoclonus" or "myoclonic" event, mentioning the word "myoclonus" or "myoclonic", Answer **No**.
- If the EEG is obscured by artifact or unable to read, Answer **No**.
- If the report contains "The patient's episodes were recorded and are not consistent with epileptic seizures.", Answer **Yes**
Below is the EEG Report: \n''',
5:f'''You are a neurologist reviewing EEG reports.
Your task is to determine the EEG information category for any non-seizure **clinical event** explicitly described in the report.

A clinical event is any clinical change such as altered mental status, confusion, fall, arrhythmia, syncope, injury, etc.
Seizures, status epilepticus should not be considered as clinical events.
Any non-accidental push button events should be considered as clinical events.
Mentioning of "typical episodes were recorded" should be considered as clinical events.
Absence of clincial events, should not be considered as clinical events. 
If the patient is under some abnormal state continuously, do not consider that state as an clinical event.

For the non-seizure clinical events described, consider the categories of EEG information for each event:
If the report includes a clear and explicit statement that there is NO EEG correlate or NO EEG change during the clinical event, using phrases such as: “No EEG correlate”, “Without EEG correlate”, “No EEG changes”, “No electrographic correlate”, **no corresponding EEG change** or other direct statement that links the event to a lack of EEG findings, the category is **False**. 
If the report describes EEG abnormalities or findings during the clinical event, the category is **True**.
If the report does not mention EEG findings during the event (the word EEG is not mentioned at all), the category is **Unclear**.
If the report described the event as **non-ictal event**, **non-epileptic event**, The category is **False** 
If the event is a "myoclonus" or "myoclonic" event, mentioning the word "myoclonus" or "myoclonic", The category is **True**.
If the EEG is obscured by artifact or unable to read, the category is **Unclear**.
If the report contains "The patient's episodes were recorded and are not consistent with epileptic seizures.", the category is **False**

After your reasoning, if there're no non-seizure clinical events, respond with **No**.
Otherwise, consider categories for all events, if there're any event whose category is "Unclear", respond with **Yes** as your FINAL ANSWER, otherwise if there's no event whose category is **Unclear** respond **No** as your FINAL ANSWER.
Below is the EEG Report: \n''',
6:f'''You are a neurologist reviewing EEG reports.
Your task is to determine whether the following report describes the patient as having **myoclonus**.
Myoclonus must be explicitly described using the word “myoclonus”,"myoclonic", or other words, phrases contains "myoclon-". DO NOT infer jerking as myoclonus.
Reasoning Instructions:
If the the report described that the patient had myoclonus, respond with **Yes**. Otherwise, respond with **No**.
Below is the EEG Report: \n''',
9:f'''You are a neurologist reviewing EEG reports.
Your task is to determine whether the following report include **any information** about **seizure** or **clinical events**.
Definitions:
**Seizure** must be explicitly described using the word **“seizure”**. Do **not** classify other EEG patterns (e.g., spikes, BIRDS, slowing, or ictal-interictal continuum) as seizures.
**Clinical event** are patient behavior, movements, or symptoms including but are not limited to limb movements, shaking, twitching, dizziness, pupil dilation, speech problems (e.g. aphasia), unresponsiveness, restlessness, tachycardia, etc.
Reasoning instructions:
If the the report does not provide any information on **seizure** or **clinical events**, respond with **Yes**. Otherwise, respond with **No**.
Steps:
1. Identify if the report has any explicit statements about **clinical event**, either that a clinical event occured or clinical events are absent.  → If Yes then Answer **No**, else continue.
2. Identify if the report has any explicit statements about **seizure**, either that a seizure occured or seizures are absent.   → If Yes then Answer **No**, else Answer **Yes**.
Below is the EEG Report: \n'''}

message_template_dict = {
0:'''Your task is to classify the report based solely on explicit statements about seizure and clinical events. 

Important constraints: 
Do not make inferences based on the contextual information. 
Do not assume a event or a seizure happened unless it is directly mentioned. 
Base your judgment only on the text presented. 

Reasoning process: 
Think step by step to interpret the note. 
Identify any ambiguities or potential sources of error. 
Reflect on how you reached your conclusion. 

Output:
After your reasoning, provide **only one word** as your final answer: **Yes**, or **No**.''',
1:'''Your task is to classify the report based solely on explicit statements about seizure and clinical events. 

Important constraints: 
- Do not make inferences based on the contextual information. 
- Do not assume a event or a seizure happened unless it is directly mentioned. 
- Base your judgment only on the text presented. 

Reasoning process: 
- Think step by step to interpret the note. 
- Identify any ambiguities or potential sources of error. 
- Reflect on how you reached your conclusion. 

Output:
After your reasoning, provide **only one word** as your final answer: **Yes**, or **No**.''',
2:'''Your task is to classify the report based solely on explicit statements about seizure and clinical events. 

Important constraints: 
Do not make inferences based on the contextual information. 
Do not assume a event or a seizure happened unless it is directly mentioned. 
Base your judgment only on the text presented. 

Reasoning process: 
Think step by step to interpret the note. 
Identify any ambiguities or potential sources of error. 
Reflect on how you reached your conclusion. 

Output:
After your reasoning, provide **only one word** as your final answer: **Yes**, or **No**.''',
3:'''Your task is to classify the report based solely on explicit statements about seizure and clinical events. 

Important constraints: 
Do not make inferences based on the contextual information. 
Do not assume a event or a seizure happened unless it is directly mentioned. 
Base your judgment only on the text presented. 

Reasoning process: 
Think step by step to interpret the note. 
Identify any ambiguities or potential sources of error. 
Reflect on how you reached your conclusion. 

Output:
After your reasoning, provide **only one word** as your final answer: **Yes**, or **No**.''',
4:'''Your task is to classify the report based solely on explicit statements about seizure and clinical events. 

Important constraints: 
Do not make inferences based on the contextual information. 
Do not assume a event or a seizure happened unless it is directly mentioned. 
Base your judgment only on the text presented. 

Reasoning process: 
Think step by step to interpret the note. 
Identify any ambiguities or potential sources of error. 
Reflect on how you reached your conclusion. 

Output:
After your reasoning, provide **only one word** as your final answer: **Yes**, or **No**.''',
5:'''Your task is to classify the report based solely on explicit statements about seizure and clinical events. 

Important constraints: 
Do not make inferences based on the contextual information. 
Do not assume a event or a seizure happened unless it is directly mentioned. 
Base your judgment only on the text presented. 

Reasoning process: 
Think step by step to interpret the note. 
Identify any ambiguities or potential sources of error. 
Reflect on how you reached your conclusion. 

Output:
After your reasoning, provide **only one word** as your final answer: **Yes**, or **No**''',
6:'''Your task is to classify the report based solely on explicit statements about myoclonus. 

Important constraints: 
Do not make inferences based on the contextual information. 
Do not assume a event or a seizure happened unless it is directly mentioned. 
Base your judgment only on the text presented. 

Reasoning process: 
Think step by step to interpret the note. 
Identify any ambiguities or potential sources of error. 
Reflect on how you reached your conclusion. 

Output:
After your reasoning, provide **only one word** as your final answer: **Yes**, or **No**.''',
9:'''Your task is to classify the report based solely on explicit statements about seizures and clinical events. 

Important constraints: 
Do not make inferences based on the contextual information. 
Do not assume a event or a seizure happened unless it is directly mentioned. 
Base your judgment only on the text presented. 

Reasoning process: 
Think step by step to interpret the note. 
Identify any ambiguities or potential sources of error. 
Reflect on how you reached your conclusion. 

Output:
After your reasoning, provide **only one word** as your final answer: **Yes**, or **No**.''',
}
n_rep = 5
for i in classes:
    pre_note_message = message_dict[i]
    message_template = message_template_dict[i]
    for n in range(n_rep):
        run_ds(df,pre_note_message, message_template, f"{output_csv_path}_class_{i}_rep_{n}_{exp_id}.csv", text_column=text_col, code_from_ans_func=code_from_ans, seed=n, max_token=max_token)

