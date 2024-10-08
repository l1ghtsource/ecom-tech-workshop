import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import os
import yaml
import plotly.graph_objects as go

INFERENCE_SCRIPT_PATH = './inference.sh'
INFERENCE_CONFIG_PATH = './configs/inference_cfg_1.yaml'
TEMP_CSV_PATH = './data/temp_input.csv'
TEMP_OUTPUT_PATH = 'output_temp.npy'
YAML_CLASSES_PATH = 'data/trends_classes.yaml'

with open(YAML_CLASSES_PATH, 'r', encoding='utf-8') as file:
    class_map = yaml.safe_load(file)['classes']

def run_inference(assessment: str, tags: str, text: str) -> np.ndarray:
    
    df = pd.DataFrame(
        {
            'index': [1],
            'assessment': [assessment],
            'tags': [tags],
            'text': [text]
        }
    )
    
    df.to_csv(TEMP_CSV_PATH, index=False)
    
    try:
        subprocess.run(
            [
                INFERENCE_SCRIPT_PATH, 
                INFERENCE_CONFIG_PATH,
                TEMP_CSV_PATH, 
                TEMP_OUTPUT_PATH
            ],
            check=True,
            shell=True
        )

        predictions = np.load(TEMP_OUTPUT_PATH)
        return predictions

    except subprocess.CalledProcessError as e:
        st.error('Ошибка при запуске инференса: ' + str(e))
        return None

    finally:
        if os.path.exists(TEMP_CSV_PATH):
            os.remove(TEMP_CSV_PATH)
        if os.path.exists(TEMP_OUTPUT_PATH):
            os.remove(TEMP_OUTPUT_PATH)

def run_demo_inference() -> np.ndarray:
    return np.random.rand(1, 50)

st.title('Множественная классификация обратной связи от пользователей')

mode = st.radio(
    'Выберите режим инференса:',
    ('Демо режим', 'Полноценный инференс')
)

assessment = st.text_input('Введите assessment')
tags = st.text_input('Введите tags')
text = st.text_area('Введите text', '', height=200)

if st.button('Запустить инференс'):
    if not assessment.strip() or not tags.strip() or not text.strip():
        st.error('Все поля (assessment, tags, text) должны быть заполнены!')
    else:
        st.write(f'Запускаем инференс модели в режиме: {mode}...')

        if mode == 'Демо режим':
            predictions = run_demo_inference()

        else:
            predictions = run_inference(assessment, tags, text)
            
        if predictions is not None:
            st.write('Инференс завершен успешно!')
            
            num_classes = 50
            class_labels = [f'Class {i}' for i in range(1, num_classes + 1)]

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=class_labels,
                y=predictions[0],
                marker=dict(color='rgba(54, 162, 235, 0.6)', line=dict(color='rgba(0, 0, 0, 1)', width=1)),
                name='Вероятности'
            ))

            fig.update_layout(
                title='Распределение вероятностей по классам',
                xaxis_title='Классы',
                yaxis_title='Вероятности',
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=True, zeroline=True),
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)


            high_prob_classes = {i: class_map[i] for i, prob in enumerate(predictions[0]) if prob >= 0.55}
            if high_prob_classes:
                st.subheader('Классы с вероятностью >= 0.55:')
                for class_id, class_name in high_prob_classes.items():
                    st.write(f"Class {class_id}: {class_name}")
            else:
                st.write('Нет классов с вероятностью >= 0.55')