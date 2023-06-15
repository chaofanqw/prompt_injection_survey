import time

import gradio as gr
import nltk
import web_config
import pandas as pd

import os
import openai
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor


def sync_kpi_function(messages, model, stream):
    response = openai.ChatCompletion.create(model=model, messages=messages, stream=stream)
    return response


# Wrapping synchronous function to run asynchronously
async def async_sync_kpi_function(messages, model, stream):
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor()
    return await loop.run_in_executor(executor, lambda: sync_kpi_function(messages, model, stream))


async def wait_for_kpi_call(max_wait_time, max_retries, messages, model, stream):
    for attempt in range(1, max_retries + 1):
        try:
            return await asyncio.wait_for(async_sync_kpi_function(messages, model, stream), timeout=max_wait_time)
        except asyncio.TimeoutError:
            logging.warning(f"KPI call timed out, retrying... (attempt {attempt}/{max_retries})")
            if attempt == max_retries:
                logging.warning("Reached maximum number of retries. Aborting.")
                return None


def chat_function(messages, gpt_3_key, gpt_4_key, model='gpt-3.5-turbo', pay=False):
    prev = time.time()

    openai.api_key = gpt_3_key if model == 'gpt-3.5-turbo' else gpt_4_key
    response = asyncio.run(wait_for_kpi_call(20, 2, messages, model, True))
    print(f"{model} has been called.")

    collected_chunks = []
    collected_messages = []

    for chunk in response:
        collected_chunks.append(chunk)  # save the event response
        chunk_message = chunk['choices'][0]['delta'].get('content', '')  # extract the message
        collected_messages.append(chunk_message)  # save the message

    return time.time() - prev, ''.join(collected_messages)


def insert_placeholder(position, message, placeholder='[attack_prompt_placeholder]'):
    placeholder = ' ' + placeholder + ' '
    message = message.replace(placeholder, '')
    if position == 'Begin':
        message = placeholder + message
    elif position == 'Middle':
        sentences = nltk.tokenize.sent_tokenize(message)
        if len(sentences) // 2 == 0:
            message = placeholder + message
        else:
            message = message.replace(sentences[len(sentences) // 2 - 1],
                                      sentences[len(sentences) // 2 - 1] + placeholder)
    elif position == 'End':
        message = message + placeholder
    return message


def manual_generation(question_type, question_content, inject_item, manual_instruction, inject_sentence_placeholder,
                      inject_item_placeholder='[inject_item_placeholder]'):
    if len(question_content) == 0 or len(inject_item) == 0:
        return '', ''

    manual_instruction = manual_instruction.replace(inject_item_placeholder, inject_item)
    question_content = question_content.replace(inject_sentence_placeholder, manual_instruction)

    if question_type == 'Close-ended':
        question_content += ' Only provide option as "Option *" without explanation.'

    return question_content, manual_instruction


def manual_evaluation(question_type, inject_position, inject_item, manual_prompt, manual_survey,
                      manual_evaluation_api, manual_evaluation_round, gpt_3_key, gpt_4_key, pay=True):
    if manual_survey == '':
        return [[]]

    history = []
    for i in range(int(manual_evaluation_round)):
        consume_time, response = chat_function([{'role': 'user', 'content': manual_survey}], gpt_3_key, gpt_4_key,
                                               model=manual_evaluation_api.lower(), pay=pay)
        history += [[question_type, inject_position, inject_item, manual_prompt, manual_survey,
                     manual_evaluation_api.lower(), consume_time, response,
                     True if inject_item.lower() in response.lower() else False]]
        summary = f"The attack prompt injected \"{inject_item}\" into {manual_evaluation_api} responses " \
                  f"for {sum([each[-1] for each in history])} out of {i + 1} evaluation round(s)."

        yield history, summary


def generate_problem_encoding(automated_instruction, automated_example, question_content, inject_item,
                              inject_item_placeholder='[inject_item_placeholder]'):
    problem_encoding = ''

    if automated_instruction != '':
        problem_encoding += '###Instruction###\n' + automated_instruction.replace(inject_item_placeholder,
                                                                                  inject_item) + '\n'

    if automated_example != '':
        problem_encoding += '###Example###\n' + automated_example + '\n'

    problem_encoding += '###Task###\nconversation:' + question_content.replace(inject_item_placeholder,
                                                                               inject_item) + '\nreplaced sentence:'

    return problem_encoding


def chatbot_interface(conversation):
    chatbot_conversation = []
    for each in conversation:
        chatbot_conversation.append((each['content'], None) if each['role'] == 'user' else (None, each['content']))

    return chatbot_conversation


def automated_generation(question_type, question_content, inject_position, inject_item, automated_instruction,
                         automated_example, automated_iteration, automated_revise, automated_metrics,
                         automated_reasoning, inject_sentence_placeholder, inject_item_placeholder, gpt_3_key,
                         gpt_4_key, automated_generation_api, automated_evaluation_api, automated_evaluation_round):
    if len(question_content) == 0 or len(inject_item) == 0:
        return {chatbot: [[]]}

    problem_encoding = generate_problem_encoding(automated_instruction, automated_example, question_content,
                                                 inject_item, inject_item_placeholder)
    evaluation_history_summary = []
    evaluation_history_summary_prompt = []

    conversation_gpt = [{'role': 'user', 'content': problem_encoding}]
    yield {chatbot: chatbot_interface(conversation_gpt)}

    for r in range(int(automated_iteration)):
        consume_time, response = chat_function(conversation_gpt, gpt_3_key, gpt_4_key,
                                               model=automated_generation_api.lower(), pay=True)

        if r != 0 and automated_reasoning != '' and 'replaced sentence:' in response.lower():
            attack_prompt = response[response.lower().index('replaced sentence:') + len('replaced sentence:'):]
        else:
            attack_prompt = response

        conversation_gpt += [{'role': 'assistant', 'content': response}]
        question_survey = question_content.replace(inject_sentence_placeholder, attack_prompt) + \
                          ('  Only provide option as "Option *" without explanation.'
                           if question_type == 'Close-ended' else '')

        yield {chatbot: chatbot_interface(conversation_gpt), automated_prompt: attack_prompt,
               automated_survey: question_survey}

        evaluation_history = []
        for i in range(int(automated_evaluation_round)):
            eval_consume_time, eval_response = chat_function([{'role': 'user', 'content': question_survey}], gpt_3_key,
                                                             gpt_4_key, model=automated_evaluation_api.lower(),
                                                             pay=True)
            evaluation_history += [[question_type, inject_position, inject_item, attack_prompt, question_survey,
                                    automated_evaluation_api.lower(), eval_consume_time, eval_response,
                                    True if inject_item.lower() in eval_response.lower() else False]]
            evaluation_history_summary.append(evaluation_history[-1])
            summary = f"The attack prompt injected \"{inject_item}\" into {automated_evaluation_api} responses " \
                      f"for {sum([each[-1] for each in evaluation_history])} out of {i + 1} evaluation round(s)."
            yield {automated_evaluation_summary: summary, automated_evaluation_result: evaluation_history_summary}

        rate = sum([each[-1] for each in evaluation_history]) / int(automated_evaluation_round)
        evaluation_history_summary_prompt.append([attack_prompt, f"{rate * 100:.1f}% ("
                                                                 f"{sum([each[-1] for each in evaluation_history])}"
                                                                 f" out of {int(automated_evaluation_round)} rounds)"])
        yield {automated_evaluation_result_summary: evaluation_history_summary_prompt}

        if r < int(automated_iteration) - 1:
            revise_information = automated_revise.replace(inject_item_placeholder, inject_item)
            revise_information += f" The previous sentence is: \"{attack_prompt}\""
            for metric in automated_metrics:
                if metric == 'Length':
                    revise_information += f", its length is: {len(attack_prompt)} (shorter is better)"
                if metric == 'Inject Rate':
                    revise_information += f", the effectiveness is: {sum([each[-1] for each in evaluation_history])} " \
                                          f"out of {len(evaluation_history)} (higher is better)"
            revise_information += '. '

            if automated_reasoning != '':
                revise_information += automated_reasoning.replace(inject_item_placeholder, inject_item)

            conversation_gpt += [{'role': 'user', 'content': revise_information}]
            yield {chatbot: chatbot_interface(conversation_gpt)}


def question_type_modification(question_type):
    manual_instruction = web_config.option['manual_instruction'] \
        if question_type == "Close-ended" else web_config.term['manual_instruction']
    automated_instruction = web_config.option['automated_instruction'] \
        if question_type == "Close-ended" else web_config.term['automated_instruction']
    automated_reasoning = web_config.option['automated_reasoning'] \
        if question_type == "Close-ended" else web_config.term['automated_reasoning']
    automated_example = web_config.option['automated_example'] \
        if question_type == "Close-ended" else web_config.term['automated_example']
    automated_revise = web_config.option['automated_revise'] \
        if question_type == "Close-ended" else web_config.term['automated_revise']

    return manual_instruction, automated_instruction, automated_reasoning, automated_example, automated_revise


if __name__ == '__main__':
    with gr.Blocks() as demo:
        gr.Markdown("**Section: API Keys**")
        with gr.Row() as key:
            gpt_3_key = gr.Textbox(label='GPT-3.5-turbo API KEY',
                                   lines=1)
            gpt_4_key = gr.Textbox(label='GPT-4 API KEY',
                                   lines=1)

        gr.Markdown("**Section: Injection Information**")
        with gr.Accordion("Placeholder Information", open=False):
            with gr.Row() as placeholder:
                inject_item_placeholder = gr.Textbox(label="Inject Option/Term Placeholder",
                                                     info="The placeholder for the inject option or term.",
                                                     value="[inject_item_placeholder]",
                                                     interactive=False)
                inject_sentence_placeholder = gr.Textbox(label="Attack Prompt Placeholder",
                                                         info="The placeholder for the attack prompt.",
                                                         value="[attack_prompt_placeholder]",
                                                         interactive=False)

        with gr.Row() as inject:
            with gr.Column():
                question_type = gr.Radio(["Close-ended", "Open-ended"],
                                         label="Survey Question Type",
                                         info="What is the question type?",
                                         value="Close-ended",
                                         interactive=True)
                inject_item = gr.Textbox(label="Inject Option/Term",
                                         info="What is the option or term you want to inject?",
                                         placeholder="Option or term away from the context of survey question.")
                inject_position = gr.Radio(["None", "Begin", "Middle", "End"],
                                           label="Attack Prompt Position",
                                           info="Where do you want to inject the attack prompt?",
                                           value="None",
                                           interactive=True)
            with gr.Column():
                question_content = gr.Textbox(label="Survey Question",
                                              info="What is the survey question?",
                                              lines=10)

        gr.Markdown("**Section: Attack Prompt Construction**")
        with gr.Row() as header:
            with gr.Tab("Manual Construction"):
                with gr.Row():
                    with gr.Column() as manual_craft:
                        manual_submit = gr.Button(value="Generate Attack Prompt")
                        manual_instruction = gr.Textbox(label="Attack Prompt Template",
                                                        info="The attack prompt's manual template. To edit the manual template, relocate '[inject_item_placeholder]' to the appropriate position.",
                                                        value=web_config.option['manual_instruction'],
                                                        interactive=True)
                        manual_prompt = gr.Textbox(label="Attack Prompt (Generated)",
                                                   interactive=False)
                        manual_survey = gr.Textbox(label="Survey Question embedded with Attack Prompt (Generated)",
                                                   lines=10,
                                                   interactive=False)

                    with gr.Column():
                        manual_evaluate_btn = gr.Button(value="Evaluate Attack Prompt")
                        manual_evaluation_api = gr.Radio(["GPT-3.5-turbo", "GPT-4"],
                                                         label="API",
                                                         info="Which API do you want to use for evaluation?",
                                                         value="GPT-3.5-turbo",
                                                         interactive=True)
                        manual_evaluation_round = gr.Number(label="Round",
                                                            info="How many rounds do you want to evaluate the attack prompt?",
                                                            value=20,
                                                            interactive=True)
                        manual_evaluation_summary = gr.Textbox(label="Evaluation Summary (Generated)",
                                                               interactive=False)

                with gr.Accordion("Evaluation Detail (Generated)", open=False):
                    manual_evaluation_result = gr.Dataframe(
                        headers=["question type", "position", "injected item", "prompt",
                                 "message", "model", "time", "response", "success"],
                        datatype=["str" for _ in range(9)],
                        wrap=True)

            with gr.Tab("Automated Construction"):
                with gr.Row():
                    with gr.Column() as automated_craft:
                        automated_generation_api = gr.Radio(["GPT-3.5-turbo", "GPT-4"],
                                                            label="API (Generation)",
                                                            info="Which API do you want to use for automated attack prompt generation?",
                                                            value="GPT-3.5-turbo",
                                                            interactive=True)

                        # gr.Markdown("**Subsection: Problem Encoding**")
                        with gr.Accordion("Subsection: Problem Encoding", open=False):
                            automated_instruction = gr.Textbox(label="Task Instruction",
                                                               info="What is the insturction for the automated process of constructing attack prompts?",
                                                               value=web_config.option['automated_instruction'],
                                                               interactive=True,
                                                               lines=3)
                            automated_example = gr.Textbox(label="Example",
                                                           info="What is the example for the automated process of constructing attack prompts?",
                                                           value=web_config.option['automated_example'],
                                                           interactive=True,
                                                           lines=4)

                        # gr.Markdown("**Subsection: Revise Attack Prompt**")
                        with gr.Accordion("Subsection: Revise Attack Prompt", open=False):
                            automated_iteration = gr.Number(label="Round (Revise)",
                                                            info="How many rounds do you want to run the automated process of constructing attack prompts?",
                                                            value=10,
                                                            interactive=True)
                            automated_revise = gr.Textbox(label="Instruction",
                                                          info="What is the instruction for the automated process to reconstruct attack prompts?",
                                                          value=web_config.option['automated_revise'],
                                                          interactive=True,
                                                          lines=2)
                            automated_metrics = gr.CheckboxGroup(["Length", "Inject Rate"],
                                                                 label="Metrics",
                                                                 info="Which metrics do you want to use for evaluating the attack prompt?",
                                                                 value=["Length", "Inject Rate"],
                                                                 interactive=True)
                            automated_reasoning = gr.Textbox(label="Chain-of-Thought Prompting",
                                                             info="What is the Chain-of-Thought Prompting for the automated process of constructing attack prompts?",
                                                             value=web_config.option['automated_reasoning'],
                                                             interactive=True,
                                                             lines=4)

                        # gr.Markdown("**Subsection: Evaluation**")
                        with gr.Accordion("Subsection: Evaluation", open=False):
                            automated_evaluation_api = gr.Radio(["GPT-3.5-turbo", "GPT-4"],
                                                                label="API (Evaluation)",
                                                                info="Which API do you want to use for evaluation?",
                                                                value="GPT-3.5-turbo",
                                                                interactive=True)
                            automated_evaluation_round = gr.Number(label="Round (Evaluation)",
                                                                   info="How many rounds do you want to evaluate the attack prompt?",
                                                                   value=20,
                                                                   interactive=True)

                        automated_generate_btn = gr.Button(value="Generate Attack Prompt")

                    with gr.Column():
                        gr.Markdown("**Attack Prompt Generation and Evaluation (Generated)**")

                        chatbot = gr.Chatbot(label="Dialogue",
                                             info="Iteration of automated process of constructing attack prompts.",
                                             interactive=False)

                        with gr.Accordion("Generated Attack Prompt under Evaluation", open=False):
                            automated_prompt = gr.Textbox(label="Attack Prompt",
                                                          interactive=False)
                            automated_survey = gr.Textbox(label="Survey Question embedded with Attack Prompt",
                                                          lines=10,
                                                          interactive=False)
                            automated_evaluation_summary = gr.Textbox(label="Evaluation Effectiveness",
                                                                      interactive=False)

                        with gr.Accordion("Evaluation Summarization over Attack Prompt", open=False):
                            automated_evaluation_result_summary = gr.Dataframe(headers=["prompt", "effectiveness"],
                                                                               datatype=["str" for _ in range(2)],
                                                                               wrap=True)

                        with gr.Accordion("Evaluation Detail", open=False):
                            automated_evaluation_result = gr.Dataframe(headers=["question type", "position",
                                                                                "injected item", "prompt", "message",
                                                                                "model", "time", "response",
                                                                                "success"],
                                                                       datatype=["str" for _ in range(9)],
                                                                       wrap=True)

        inject_position.change(insert_placeholder,
                               [inject_position, question_content, inject_sentence_placeholder],
                               question_content)
        question_type.change(question_type_modification,
                             [question_type],
                             [manual_instruction, automated_instruction, automated_reasoning, automated_example,
                              automated_revise])
        manual_submit.click(manual_generation,
                            [question_type, question_content, inject_item, manual_instruction,
                             inject_sentence_placeholder],
                            [manual_survey, manual_prompt])
        manual_evaluate_btn.click(manual_evaluation,
                                  [question_type, inject_position, inject_item, manual_prompt, manual_survey,
                                   manual_evaluation_api, manual_evaluation_round, gpt_3_key, gpt_4_key],
                                  [manual_evaluation_result, manual_evaluation_summary])
        automated_generate_btn.click(automated_generation,
                                     [question_type, question_content, inject_position, inject_item,
                                      automated_instruction,
                                      automated_example, automated_iteration, automated_revise, automated_metrics,
                                      automated_reasoning, inject_sentence_placeholder, inject_item_placeholder,
                                      gpt_3_key,
                                      gpt_4_key, automated_generation_api, automated_evaluation_api,
                                      automated_evaluation_round],
                                     [chatbot, automated_prompt, automated_survey, automated_evaluation_summary,
                                      automated_evaluation_result, automated_evaluation_result_summary])

    demo.queue().launch()
