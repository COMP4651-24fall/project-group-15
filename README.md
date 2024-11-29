# COMP4651 Project -- An IPFS-Blockchain-based Decentralized Federated Learning System for E-Commerce


## Introduction
In training a deep learning model, accessing a large enough dataset is always a challenge. In e-commerce, it is difficult for a small enterprise to acquire a large enough data sample to train an accurate recommendation system. While pooling data from multiple e-commerce clients to train a centralized model could enhance recommendation performance, such an approach is infeasible due to privacy concerns. In recent years the Federated Learning (FL) system has emerged as an alternative. Participating entities train models on their local dataset and share updates. However this framework relies on a trusting centralized server. To address these limitations, decentralized FL frameworks have been proposed, leveraging peer-to-peer interactions to eliminate the need for a central coordinator.

We propose an IPFS-Blockchain-based Distributed Decentralized Federated Learning System to facilitate cross-merchant recommendation model training. By leveraging federated learning (FL) systems, combined with blockchain, distributed storage systems, and cloud computing via Function as a Service provider (FaaS), this system provides privacy-protected collaboration and scalability while maintaining robustness. This project primarily focuses on the realization of distributed and parallel collaboration over the Internet rather than AI model optimization.

# Code Descriptions 

## [Ethereum Smart Contract](eth_smart_contract/store_multiple_hash.sol)
* It stores the CID hashes of trained models stored on IPFS by each client
* This smart contract is published on Ethereum's *Sepolia Testnet*
* Address is 0x980B6A9D39AdbDA1435b8498C429A19e04466237
* Transactions can be viewed on [Etherscan](https://sepolia.etherscan.io/address/0x980b6a9d39adbda1435b8498c429a19e04466237)

## [IPFS Handler](lambda/ipfs_storage.js)
* A lambda handler for file read and write to IPFS
* The IPFS is initiated in the EC2 VM as mentioned below
* All file content should be transformed into **base64 encoded** before uploading
* Sample of file reading via HTTP POST in python
    ```
    url = "https://2zk0vq0ll6.execute-api.us-east-1.amazonaws.com/default/ipfs-file-streaming"
    payload = {
        "action": "1",
        "cid": "<CID of that file>"
    }
    response = requests.post(url, json=payload)
    response_data = response.json()
    ```
    > **Note :** Set `action` to "1" for reading and "0" for writing
* Response
    ```
    {   
        "status": 'true',
        "cid": "<CID of the hash>",
        "fileContent" : "<File content in base64 encoded>"
    }
    ```
* Sample of file uploading via HTTP POST in python
    ```
    url = "https://2zk0vq0ll6.execute-api.us-east-1.amazonaws.com/default/ipfs-file-streaming"
    payload = {
        "action": "0",
        "fileName": "<File name>"
        "fileContent" : "<File content in base64 encoded>"
    }
    response = requests.post(url, json=payload)
    response_data = response.json()
    ```
* Response
    ```
    {   
        "status": 'true',
        "cid": "<CID of the hash>"
    }
    ```

## [Blockchain Handler](lambda/blockchain-storage.py)
* A lambda handler for uploading and retrieveing hash to and from ETH smart contract
* Digital wallet hard-coded in the function
    > **Note I :** Please **DO NOT** disclose or steal my wallets! It's definitely not a NO to expose the private key of wallet to public but given the TAs may try out the system, I still included here.
    
    > **Note II :** Although the cryptocurrency used here is Sepolia ETH, which is basically *fake* ETH, still, please **DO NOT** spam and use all of it.

* Sample hash reading via HTTP POST in python
    ```
    url = "https://82o2i4mwfc.execute-api.us-east-1.amazonaws.com/default/blockchain-storage"
    payload = {
        "action": "1",
        "clientIndex": "<Client index>",
    }
    response = requests.post(url, json=payload)
    ```
    > **Note :** Set `action` to "1" for reading and "0" for writing
* Response
    ```
    {
        'status': 'true',
        'hash': "<hash of the client>",
    }
    ```
* Sample hash uploading via HTTP POST in python
    ```
    url = "https://82o2i4mwfc.execute-api.us-east-1.amazonaws.com/default/blockchain-storage"
    payload = {
        "action": "1",
        "clientIndex": "<Client index>",
        "content" : "<hash to be uploaded>"
    }
    response = requests.post(url, json=payload)
    ```
## [Federated Learning Pipeline Handler](lambda/ipfs-blockchain-federated-pipeline.py)
* A lambda handler for the whole pipeline
* This handler perform the IPFS-Blockchain storage and model aggregation
* Client should first trained a new model with their own dataset locally, then in this handler, we upload the **base64 encoded** model to IPFS and update the smart contract, then initiate [model aggregation handler](lambda/federated-learning-gcp.py) for the other 2 clients parallelly and distributedly
* Noted that the aggregation handler is deployed on Google Cloud Platform cloud run function as we found that it is complicated to add pytorch packages onto AWS lambda, so we used an alternative FaaS
* The aggregation is performed by averaging the weights across models as a simple demo purpose (There are more ways to do so e.g. FedAvg, but here we make the system minimal due to time limitation)
* Then the new models are updated in the IPFS and blockchain storage system
* Sample HTTP POST request in python
    ```
    url = "https://g6jentj8ia.execute-api.us-east-1.amazonaws.com/default/ipfs-blockchain-federated-pipeline"
    payload = {
        "updatedClientIndex": "<Index of client updated their model>",
        "blockchainEnabled": "1",
        "modelEncoded": "<File content in base64 encoded>"
    }
    response = requests.post(url, json=payload)
    ```
    > **Note :** `blockchainEnabled` is for debugging purpose <code>"0"</code> for debugging so that it will upload NOT to blockchain to save cryptocurrency
* Response
    ```
    {
        "statusCode": 200,
        "body": "true"
    }
    ```

# How to Use

## Frontend UI
1.  run `pip install Flask pandas` in the terminal
2.  run the command `python app.py` in the terminal

## Start the IPFS Server
1. Download our AWS key pair from [./EC2_instance/4651-project-keypair.pem](./EC2_instance/4651-project-keypair.pem)
    > **Note:** Make sure to save the file in the appropriate directory where you want to work with it.
2. Navigate to the directory \
    Open your terminal and navigate to the directory where the file is saved. For example
3. Run `ssh -i "4651-project-keypair.pem" ubuntu@ec2-54-174-31-161.compute-1.amazonaws.com`\
    This will connect to our EC2 instance, and the IPFS server will be automatically iniiated

## Whole Process
...describe how to get recommendation, contribute



