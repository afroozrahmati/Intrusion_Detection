# Intrusion_Detection
Building a Robust Federated learning based Anomaly Detection in Internet of Things

 

The Internet of Things (IoT) has emerged as the next big technological revolution in recent years with the potential to transform every sphere of human life. As devices, applications, and communication networks become increasingly connected and integrated, security and privacy concerns in IoT are growing at an alarming rate as well. While existing research has largely focused on centralized systems to detect security attacks, these systems do not scale well with the rapid growth of IoT devices and pose a single-point of failure risk. Furthermore, since data is extensively dispersed across huge networks of connected devices, decentralized computing is critical. 

 

Federated learning (FL) systems in the recent times has gained popularity as the distributed machine learning model that enables IoT edge devices to collaboratively train models in a decentralized manner while ensuring that data on a user’s device stays private without the contents or details of that data ever leaving that device. Federated learning is however also increasingly subject to poisoning attacks, where a malicious user can intentionally sabotage the model either by participating in the training process with mislabeled data or by directly contributing model updates which are otherwise intended to harm the performance of the shared model.

 

In this project, we propose a federated learning based anomaly detection system using LSTM Autoencoder. The proposed technique allows IoT devices to train a global model without revealing their private data, enabling the training model to grow in size while protecting each participant's local data. Further to mitigate poisoning attacks on FL, we also propose a defense mechanism, Area Similarity FoolsGold to better identify similarities between participants and mitigate adversaries who share a malicious objective. We aim to conduct extensive experiments using the Bot-IOT data set and demonstrate that our solution can not only effectively improve IoT security against unknown attacks but also ensure user’s data privacy
