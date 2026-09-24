from google.cloud import aiplatform
import pandas as pd

project_id = "hs-social-interaction-lab"  
location = "us-south1"        
endpoint_id = "mg-endpoint-3249fb88-e14e-403a-b015-93cf2368b678" 

aiplatform.init(project=project_id, location=location)
endpoint = aiplatform.Endpoint(endpoint_id)

df = pd.read_csv("../../data/final_pairs_ldp_cabnc.csv")
prepend_prompt = "You are doing a task in which you rate how common it is for certain things to have certain features. You respond out of the following options: Never, Rarely, Sometimes, About half the time, Often, Almost always, or Always.\n"

responses = ("never", "rarely", "sometimes", "about half the time", "often", "almost always", "always")

def get_rating(adjective, noun, adj_article, article):
	if article == "NA" or not article or type(article) != str:
		question = "How common is it for " + noun + " to be " + adjective + " " + noun + "?\n\nAnswer:"
	else:
		question = "How common is it for " + article + " " + noun + " to be " + adj_article + " " + adjective + " " + noun + "?\n\nAnswer:"
	response = endpoint.predict(
		instances = [{"prompt": prepend_prompt + " " + question}]
		)
	response = response.predictions[0]
	print(response)
	response = response[len(prepend_prompt + " " + question):].strip()
	for prefix in ["Answer:", "Output:", "Response:"]:
		if response.startswith(prefix):
			response = response[len(prefix):].strip()
	response = response.partition('\n')[0].lower().strip()
	response = response.partition('.')[0].lower().strip()
	response = response.partition('!')[0].lower().strip()
	response = response.partition(',')[0].lower().strip()
	
	if response not in responses:
		print("Failed with response: " + response)
		return get_rating(adjective, noun, adj_article, article)
	else:
		return response

df['llama_405b_instruct_judgment'] = df.apply(lambda x: get_rating(x['adjective'], x['noun'], x['adj_article'], x['article']).strip(), axis = 1)
df.to_csv('../../data/llama_3.1_405b_instruct_judgments.csv', index = None)