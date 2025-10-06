# Kubernetes Deployment Guide

This guide will walk you through deploying the application using Kubernetes.

## Prerequisites

- A running Kubernetes cluster.
- `kubectl` configured to connect to your cluster.

## Deployment Steps

1.  **Deploy MongoDB:**

    First, you need to deploy the MongoDB database. Before you do that, you need to add the credentials to the `mongo-secret.yml` file. The values for `mongo-user` and `mongo-password` should be base64 encoded.

    To encode your username and password, you can use the following commands:
    ```bash
    echo -n 'your-username' | base64
    echo -n 'your-password' | base64
    ```

    Update the `kubernetes/database/mongo-secret.yml` with the encoded values.

    Then, apply the Kubernetes manifests for MongoDB:
    ```bash
    kubectl apply -f kubernetes/database/mongo-config.yml
    kubectl apply -f kubernetes/database/mongo-secret.yml
    kubectl apply -f kubernetes/database/mongo.yml
    ```

2.  **Deploy the Backend:**

    Before deploying the backend, you need to update the `kubernetes/backend-deployment.yml` file with the correct MongoDB username and password. Replace `<user>` and `<password>` with the plain text values.

    Then, apply the Kubernetes manifests for the backend:
    ```bash
    kubectl apply -f kubernetes/backend-deployment.yml
    kubectl apply -f kubernetes/backend-service.yml
    ```

3.  **Verify the Deployment:**

    You can check the status of your deployments by running:
    ```bash
    kubectl get deployments
    kubectl get services
    kubectl get pods
    ```

    You should see the `mongo-deployment` and `backend-deployment` running, along with their corresponding services and pods.
