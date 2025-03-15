import transformers as tr

amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)

user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(
	scores,
	results,
	kFactor = 4,
) {
	for (const result of results) {
		const { first, second, outcome } = result;
		const firstScore = scores[first] ?? 1000;
		const secondScore = scores[second] ?? 1000;

		const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
		const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
		let sa = 0.5;
		if (outcome === 1) {
			sa = 1;
		} else if (outcome === -1) {
			sa = 0;
		}
		scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
		scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
	}
	return scores;
}\n```"""

prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_message},
    ],
    add_generation_prompt=True,
    tokenize=False,
)


def contrastive_generation(amateur, expert, prompt, max_tokens) -> str:
    '''Implements https://arxiv.org/abs/2210.15097'''

    # Tokenize the prompt
	prompt = tokenizer(prompt)["input_ids"]

	# Use GPU if available
	device = torch.device(
		"mps" if torch.backends.mps.is_available() # (I'm coding on a Mac w/ an M1 chip)
		else "cuda" if torch.cuda.is_available() # (I tested using Google Colab)
		else "cpu"
	)
	amateur = amateur.to(device)
	expert = expert.to(device)

	# Generate response
	response = []
	for _ in range(max_tokens):
		print(_)
		# Get amateur logits and expert logits
		input_tokens = torch.tensor([prompt + response]).to(device) # (1, input_length)
		amateur_logits = amateur(input_tokens).logits[0,-1,:]
		expert_logits = expert(input_tokens).logits[0,-1,:]
		
		# Get the next token using CD objective
		next_token = torch.argmax(expert_logits - amateur_logits).item()
		if next_token == tokenizer.eos_token_id:
			break
		response.append(next_token)

	return tokenizer.decode(response)

amateur = tr.AutoModelForCausalLM.from_pretrained(amateur_path)
expert = tr.AutoModelForCausalLM.from_pretrained(expert_path)
response = contrastive_generation(amateur, expert, prompt, 100)
print(response)