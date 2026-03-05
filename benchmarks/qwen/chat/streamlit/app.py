#!/usr/bin/env python3
"""
================================================================================
PROJECT: VLM Learn / ParkCircus Productions 🚀
VERSION: 1.0.1
UPDATED: 2026-03-05 08:13:21
COPYRIGHT: (c) 2026 ParkCircus Productions; All Rights Reserved.
AUTHOR: Matha Goram
LICENSE: MIT
PURPOSE: [REPLACE WITH FILE DESCRIPTION]
================================================================================
"""
# # -*- coding: utf-8 -*-
"""
A simple Streamlit app that uses a Hugging Face Transformers pipeline to classify images
as "hot dog" or "not hot dog" using a pre-trained model.
@see https://huggingface.co/julien-c/hotdog-not-hotdog
@see https://streamlit.io/generative-ai
@note The use_column_width parameter has been deprecated and
will be removed in a future release.
Please utilize the use_container_width parameter instead.
"""
import streamlit as st
from transformers import pipeline
from PIL import Image

pipeline = pipeline(task="image-classification", model="julien-c/hotdog-not-hotdog")

st.title("Hot Dog? Or Not?")

file_name = st.file_uploader("Upload a hot dog candidate image")

if file_name is not None:
    col1, col2 = st.columns(2)

    image = Image.open(file_name)
    col1.image(image, use_column_width=True)
    predictions = pipeline(image)

    col2.header("Probabilities")
    for p in predictions:
        col2.subheader(f"{ p['label'] }: { round(p['score'] * 100, 1)}%")
