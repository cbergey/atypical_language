from huggingface_hub import InferenceClient
import os
import pandas as pd

client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
)

def query_llama_instruct(prompt):
    output = client.chat_completion(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=15,
        temperature=0.6,
    )
    return output.choices[0].message.content

df = pd.read_csv("../../data/final_pairs_ldp_cabnc.csv")
df_len = round(len(df.index)/2)
df = df[:df_len]
prepend_prompt = "You are doing a task in which you rate how common it is for certain things to have certain features. You respond out of the following options: Never, Rarely, Sometimes, About half the time, Often, Almost always, or Always.\n"

responses = ("never", "rarely", "sometimes", "about half the time", "often", "almost always", "always")

def get_rating(adjective, noun, adj_article, article):
	if article == "NA" or not article or type(article) != str:
		question = "How common is it for " + noun + " to be " + adjective + " " + noun + "?"
	else:
		question = "How common is it for " + article + " " + noun + " to be " + adj_article + " " + adjective + " " + noun + "?"
	print(question)
	response = query_llama_instruct(
		prompt=prepend_prompt + " " + question
		)
	response = response.partition('.')[0].lower().strip()
	print(response)
	if response not in responses:
		print("Failed with response: " + response)
		return get_rating(adjective, noun, adj_article, article)
	else:
	    return response

df['llama_base_judgment'] = df.apply(lambda x: get_rating(x['adjective'], x['noun'], x['adj_article'], x['article']).strip(), axis = 1)
df.to_csv('../../data/llama_instruct_judgments_1.csv', index = None)