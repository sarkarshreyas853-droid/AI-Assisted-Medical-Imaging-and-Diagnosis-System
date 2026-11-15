# Limitations and bias

- **Not a medical device.** This project is for education. It must not be used for diagnosis, triage or treatment.
- **Dataset bias.** Public chest X-ray datasets come from a small number of hospitals and populations. Performance may not transfer to other scanners, patients or regions.
- **Label noise.** Labels in public datasets are often generated automatically from radiology reports and can be wrong.
- **Class balance.** Pneumonia and normal cases may not reflect real-world prevalence, so reported accuracy can be misleading. Check AUC, sensitivity and specificity by subgroup.
- **Explanations are not proof.** Grad-CAM highlights regions that influenced the model, but a highlighted region is not evidence of disease.
- **Segmentation data.** The spleen dataset is a small research dataset and does not represent clinical imaging workflows.
- **Security and privacy.** Uploaded images are kept in the audit log only as metadata. Do not upload real patient data to this demo.
