---
layout: section
---

# Course Outline

---
level: 2
---

# Welcome to "Introduction to PyTorch"

<br></br>

### July 2025

📍 Location: Lugano
<br></br>
👨‍🏫 Instructors: Rafael Sarmiento, Rocco Meli, Fabian Bösch
<br></br>
🔗 Hosted by: CSCS / ETH Zurich

---
level: 2
---

## What to Expect

This workshop is hands-on and beginner-friendly.
We assume:
- Familiarity with Python
- None to little exposure to basic ML concepts

<br></br>

## Logistics & Communication

- Questions? Ask anytime!
  - We also have a Slack channel
- Course materials: [https://github.com/eth-cscs/pytorch-training](https://github.com/eth-cscs/pytorch-training)
- We'll use CSCS' ALPS cluster for the exercises
  - Login details will be provided (e-mail)

---
level: 2
---

## Course Overview

We’ll cover:

- Fundamentals of Machine Learning
- Working with PyTorch
- Hands on exercises with JupyterLab on CSCS' ALPS cluster
- Feature extraction from text and images
- Convolutional Neural Networks (CNNs)
- Distributed training
- Introduction to Large Language Models (LLMs)

<Admonition type="info" title="Course Material">
Link to the course material and exercises can be found at: https://github.com/eth-cscs/pytorch-training
</Admonition>

---
level: 2
---

## Timetable

|             | Wednesday                         | Thursday                        | Friday                        |
|-------------|-----------------------------------|---------------------------------|-------------------------------|
| **Start**   | 10:00                            | 09:00                           | 09:00                         |
| **Morning** | Introduction to Neural <br> Networks | More on Training     | Introduction to LLMs               |
| **Lunch**   | 12:00–13:00                       | 12:00–13:00                      | 12:00–13:00                   |
| **Afternoon** | Introduction to CNNs & <br> Feature extraction for <br> computer vision and <br> language modeling | Machine Room visit <br> Distributed training | LLMs (continued)              |
| **End**   | 17:00                            | 17:00                           | 16:00                         |

---
level: 2
---

## Credentials for accessing the HPC systems at CSCS

- Course accounts `classxxx` will be provided to you via email
- These accounts are valid only for the duration of the course
    - Please copy any changes in the repository to your laptop before the end of the course


<br></br>

## Course Reservations

- We have reserved a set of nodes for the course
- Access them using the Slurm reservation `pytorch`
- The Slurm account for our course is `crs01`

---
level: 2
---

# JupyterLab

<div grid="~ cols-2 gap-4">
<div class="col-span-1">

- We will be using [https://jupyter-daint.cscs.ch](https://jupyter-daint.cscs.ch)
- You can login using your credentials provided via email
- Once you logged in
    - you will find the button "Start my server" (see image)
    - Select the container image labeled as "(course)"
    - Set the job duration to 8 hours
    - Click on "Advanced options" and type "pytorch" on the "Reservation" field
    - Launch JupyterLab

</div>
<div class="col-span-1">

<div class="w-full">
        <img src="/images/jupyterhub-launcher-pytorch2025.png" />
</div>

</div>
</div>

---
level: 2
---

# SSH Access to CSCS Cluster

<div grid="~ cols-12 gap-4">
<div class="col-span-4">

- For a convenience, it is recommended to add the following lines to your `~/.ssh/config` file

```
Host ela
    HostName ela.cscs.ch
    User <YOUR_USERNAME>
    ForwardAgent yes
    ForwardX11 yes

Host daint
    HostName daint.cscs.ch
    User <YOUR_USERNAME>
    ProxyJump ela
    ForwardAgent yes
    ForwardX11 yes
```

</div>
<div class="col-span-7">


- After that, you can connect to the cluster using the command below

```bash
$ ssh <YOUR_USERNAME>@daint
```

- You should be dropped into the `bash` shell in your home directory
- For launching jobs and we use the scratch directory

```bash
$ cd $SCRATCH
```

- here you can git-clone the course material

```bash
$ git clone https://github.com/eth-cscs/pytorch-training.git
```



</div>
</div>
