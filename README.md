# gtc2017-labs

Labs I attended at GTC 2017.
# gtc-2017-labs

## Labs I attended

### S7699 - AN INTRODUCTION TO CUDA PROGRAMMING PRESENTED BY ACCELEWARE (SESSION 1 OF 4)

Chris Mason - Technical Product Manager, Acceleware Ltd.

Join us for an informative introductory tutorial intended for those new to CUDA and which serves as the foundation for our following three tutorials. Those with no previous CUDA experience will leave with essential knowledge to start programming in CUDA. For those with previous CUDA experience, this tutorial will refresh key concepts required for subsequent tutorials on CUDA optimization. The tutorial will begin with a brief overview of CUDA and data-parallelism before focusing on the GPU programming model. We'll explore the fundamentals of GPU kernels, host and device responsibilities, CUDA syntax, and thread hierarchy. We'll deliver a programming demonstration of a simple CUDA kernel. We'll also provide printed copies of the material to all attendees for each session - collect all four!

```
Additional Session Information
RECORDING:http://on-demand.gputechconf.com/gtc/2017/video/s7699-mason-chris-an_introduction_to_cuda.mp4
PDF:Coming Soon
AUDIENCE LEVEL:Beginner
SESSION TYPE:Tutorial
ALL TOPICS:Other, Programming Languages
INDUSTRY SEGMENTS:Consulting Services
SESSION LENGTH:1h 20mSession Schedule
Monday, May 8, 9:00 AM - 10:20 AM
– Marriott Ballroom 3
```

### S7700 - AN INTRODUCTION TO THE GPU MEMORY MODEL - PRESENTED BY ACCELEWARE (SESSION 2 OF 4)

Chris Mason - Technical Product Manager, Acceleware Ltd.

This tutorial is for those with a basic understanding of CUDA who want to learn about the GPU memory model and optimal storage locations. Attend session 1, "An Introduction to GPU Programming," to learn the basics of CUDA programming that are required for Session 2. We'll begin with an essential overview of the GPU architecture and thread cooperation before focusing on different memory types available on the GPU. We'll define shared, constant, and global memory, and discuss the best locations to store your application data for optimized performance. We'll deliver a programming demonstration of shared and constant memory. We'll also provide printed copies of the material to all attendees for each session ? collect all four!

```
Additional Session Information
RECORDING:http://on-demand.gputechconf.com/gtc/2017/video/s7700-mason-chris_an_introduction_to_the_gpu.mp4
PDF:Coming Soon
AUDIENCE LEVEL:Beginner
SESSION TYPE:Tutorial
ALL TOPICS:Programming Languages
INDUSTRY SEGMENTS:Consulting Services
SESSION LENGTH:1h 20mSession Schedule
Monday, May 8, 10:30 AM - 11:50 AM
– Marriott Ballroom 3
```

### S7705 - ASYNCHRONOUS OPERATIONS AND DYNAMIC PARALLELISM IN CUDA - PRESENTED BY ACCELEWARE (SESSION 3 OF 4)

Chris Mason - Technical Product Manager, Acceleware Ltd.

This tutorial builds on the two previous sessions ("An Introduction to GPU Programming" and "An Introduction to GPU Memory Model") and is intended for those with a basic understanding of CUDA programming. This tutorial dives deep into asynchronous operations and how to maximize throughput on both the CPU and GPU with streams. We'll demonstrate how to build a CPU/GPU pipeline and how to design your algorithm to take advantage of asynchronous operations. In the second part of the session, we'll focus on dynamic parallelism. We'll deliver a programming demo involving asynchronous operations. We'll also provide printed copies of the material to all attendees for each session - collect all four!

```
Additional Session Information
RECORDING:Coming Soon
PDF:Coming Soon
AUDIENCE LEVEL:All
SESSION TYPE:Tutorial
ALL TOPICS:Programming Languages, Other
INDUSTRY SEGMENTS:Consulting Services
SESSION LENGTH:1h 20mSession Schedule
Monday, May 8, 1:00 PM - 2:20 PM
– Marriott Ballroom 3
```

### S7706 - ESSENTIAL CUDA OPTIMIZATION TECHNIQUES - PRESENTED BY ACCELEWARE (SESSION 4 OF 4)

Chris Mason - Technical Product Manager, Acceleware Ltd.

This tutorial is for those with some background in CUDA, including an understanding of the CUDA memory model and streaming multiprocessor. Our previous three tutorials provide the background information necessary for this session. This informative tutorial will provide an overview of the analysis performance tools and key optimization strategies for compute, latency, and memory bound problems. The session will include techniques for ensuring peak utilization of CUDA cores by choosing the optimal block size. It'll also include code examples and a programming demonstration highlighting the optimal global memory access pattern applicable to all GPU architectures. We'll provide printed copies of the material to all attendees for each session ? collect all four!

```
Additional Session Information
RECORDING:Coming Soon
PDF:Coming Soon
AUDIENCE LEVEL:All
SESSION TYPE:Tutorial
ALL TOPICS:Programming Languages, Other
INDUSTRY SEGMENTS:Consulting Services
SESSION LENGTH:1h 20mSession Schedule
Monday, May 8, 2:30 PM - 3:50 PM
– Marriott Ballroom 3
```

### S7634 - BUILD A NEURAL TRANSLATION SYSTEM FROM SCRATCH WITH PYTORCH

Jeremy Howard - Entrepreneur, fast.ai

As recently covered by the New York Times, Google has totally revamped its Translate tool using deep learning. We'll learn about what's behind this system, and similar state of the art systems?including some more recent advances that haven't yet found their way into Google's tool. We'll start with looking at the original encoder-decoder model that neural machine translation is based on, and will discuss the various potential applications of this kind of sequence to sequence algorithm. We'll then look at attentional models, including applications in computer vision (where they are useful for large and complex images). In addition, we'll investigate stacking layers, both in the form of bidirectional layers and deep RNN architectures. We'll focus on the practical details of training real-world translation systems, and showing how to take advantage of PyTorch's dynamic nature to heavily customize an RNN as required for modern translation approaches.

```
Additional Session Information
RECORDING:Coming Soon
PDF:Coming Soon
AUDIENCE LEVEL:Advanced
SESSION TYPE:Tutorial
ALL TOPICS:Deep Learning and AI
INDUSTRY SEGMENTS:General
SESSION LENGTH:50 minutesSession Schedule
Monday, May 8, 5:00 PM - 5:50 PM
– Grand Ballroom 220A
```

### L7122 - IMAGE SEGMENTATION WITH TENSORFLOW (PRESENTED BY NVIDIA DEEP LEARNING INSTITUTE)

Jonathan Bentz - Solutions Architect, NVIDIA

There are a variety of important applications that need to go beyond detecting individual objects within an image, and that instead need to segment the image into spatial regions of interest. An example of image segmentation involves medical imagery analysis, where it is often important to separate the pixels corresponding to different types of tissue, blood or abnormal cells, so that you can isolate a particular organ. Another example includes self-driving cars, where segmenting an image into distinct areas is needed to understand road scenes. In this lab, you will learn how to train and evaluate an image segmentation network using TensorFlow. Prerequisites: Basic knowledge of TensorFlow. This lab utilizes GPU resources in the cloud, you are required to bring your own laptop.

```
Additional Session Information
RECORDING:
PDF:
AUDIENCE LEVEL:Intermediate
SESSION TYPE:Instructor-Led Lab
ALL TOPICS:Deep Learning and AI
INDUSTRY SEGMENTS:Other
SESSION LENGTH:2 hoursSession Schedule
Tuesday, May 9, 9:30 AM - 11:30 AM
– Room LL20
```

### L7135 - DEEP LEARNING FOR MEDICAL IMAGE ANALYSIS USING R AND MXNET (PRESENTED BY NVIDIA DEEP LEARNING INSTITUTE)

Charles Killam - Curriculum Designer & Certified Instructor, NVIDIA
Abel Brown, NVIDIA

Convolutional neural networks (CNNs) have proven to be just as effective in visual recognition tasks involving non-visible image types as regular RGB camera imagery. One important application of these capabilities is medical image analysis, where we wish to detect features indicative of medical conditions and use them to infer patient status. In addition to processing non-visible imagery, such as CT scans and MRI, these applications often require us to process higher dimensionality imagery that may be volumetric and have a temporal component. In this lab you will use the deep learning framework MXNet to train a CNN to infer the volume of the left ventricle of the human heart from a time-series of volumetric MRI data. You will learn how to extend the canonical 2D CNN to be applied to this more complex data and how to directly predict the ventricle volume rather than generating an image classification. In addition to the standard Python API, you will also see how to use MXNet through R, which is an important data science platform in the medical research community. Prerequisites: Basic knowledge of CNNs. This lab utilizes GPU resources in the cloud, you are required to bring your own laptop.

```
Additional Session Information
RECORDING:
PDF:
AUDIENCE LEVEL:Intermediate
SESSION TYPE:Instructor-Led Lab
ALL TOPICS:Deep Learning and AI, AI in Healthcare Summit
INDUSTRY SEGMENTS:Healthcare & Life Sciences
SESSION LENGTH:2 hoursSession Schedule
Tuesday, May 9, 1:30 PM - 3:30 PM
– Room LL21E
```

### L7126 - DEEP LEARNING FOR IMAGE AND VIDEO CAPTIONING (PRESENTED BY NVIDIA DEEP LEARNING INSTITUTE)

Allison Gray - Solutions Architect, NVIDIA

Effective descriptions of content within images and video clips has been performed with convolutional and recurrent neural networks. Attendees will apply a deep learning technique via a framework to create captions on data and generate their own captions. Prerequisite: Familiarity with deep learning and a framework. This lab utilizes GPU resources in the cloud, you are required to bring your own laptop.

```
Additional Session Information
RECORDING:
PDF:
AUDIENCE LEVEL:Intermediate
SESSION TYPE:Instructor-Led Lab
ALL TOPICS:Deep Learning and AI
INDUSTRY SEGMENTS:General, Media & Entertainment, Internet / Telecommunications, Higher Education / Research
SESSION LENGTH:4 hoursSession Schedule
Wednesday, May 10, 12:00 PM - 4:00 PM
– Room LL20
```

### L7115 - IN-DEPTH PERFORMANCE ANALYSIS FOR OPENACC/CUDA/OPENCL APPLICATIONS WITH SCORE-P AND VAMPIR

Robert Henschel - Director Science Community Tools, Indiana University
Jiri Kraus - Senior Devtech Compute, NVIDIA
Guido Juckeland - Head of Computational Science Group, Helmholtz-Zentrum Dresden-Rossendorf

Work with Score-P/Vampir to learn how to dive into the execution properties of CUDA and OpenACC applications. We'll show how to use Score-P to generate a trace file and how to study it with Vampir. Additionally, we'll use the newly established OpenACC tools interface to present how OpenACC applications can be studied for performance bottlenecks. This lab uses GPU resources in the cloud, so bring your laptop. Prerequisites: Basic knowledge on CUDA/OpenACC and MPI is recommended but not required. This lab utilizes GPU resources in the cloud, you are required to bring your own laptop.

```
Additional Session Information
RECORDING:
PDF:
AUDIENCE LEVEL:Intermediate
SESSION TYPE:Instructor-Led Lab
ALL TOPICS:Tools and Libraries, HPC and Supercomputing
INDUSTRY SEGMENTS:General
SESSION LENGTH:2 hoursSession Schedule
Wednesday, May 10, 4:00 PM - 6:00 PM
– Room LL21A
```
