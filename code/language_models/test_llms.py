from huggingface_hub import InferenceClient
import os
import pandas as pd


client = InferenceClient(
    model="meta-llama/Llama-3.1-70B",
)

def query_llama_base(prompt):
    output = client.text_generation(
        prompt,
        max_new_tokens=15,
        temperature=0.6,
        do_sample=True,
        return_full_text=False,
    )
    return output

df = pd.read_csv("../../data/final_pairs_ldp_cabnc.csv")
df_len = round(len(df.index)/3)
df = df[:10]
prepend_prompt = "You are doing a task in which you rate how common it is for certain things to have certain features. You respond out of the following options: Never, Rarely, Sometimes, About half the time, Often, Almost always, or Always.\n"

responses = ("never", "rarely", "sometimes", "about half the time", "often", "almost always", "always")

def get_rating(adjective, noun, adj_article, article):
	if article == "NA" or not article or type(article) != str:
		question = "How common is it for " + noun + " to be " + adjective + " " + noun + "?\n\nAnswer:"
	else:
		question = "How common is it for " + article + " " + noun + " to be " + adj_article + " " + adjective + " " + noun + "?\n\nAnswer:"
	print(question)
	response = query_llama_base(
		prompt=prepend_prompt + " " + question
		)
	response = response.partition('\n')[0].lower().strip()
	if response not in responses:
		print("Failed with response: " + response)
		return get_rating(adjective, noun, adj_article, article)
	else:
	    return response

df['llama_base_judgment'] = df.apply(lambda x: get_rating(x['adjective'], x['noun'], x['adj_article'], x['article']).strip(), axis = 1)
df.to_csv('../../data/llama_base_judgments_1.csv', index = None)