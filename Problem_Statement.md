## Problem Statement Background

In industrial facilities and laboratory environments, enforcing safety compliance such as
wearing protective helmets/caps, safety shoes, and protective eyewear (goggles/specs)
is critical to prevent accidents, injuries, and operational hazards.
Despite the importance of these safety measures, compliance monitoring is still largely
manual or dependent on periodic inspections. These approaches are:
● Labour-intensive
● Inconsistent across shifts and locations
● Difficult to scale in high-traffic environments
● Prone to human error
Conventional CCTV systems lack intelligence and cannot automatically detect whether
safety rules are being followed. Many existing AI solutions operate on static images, lack
real-time capabilities, or fail to associate violations with specific individuals for tracking
and auditing.
There is a strong need for an AI-driven computer vision system that can automatically
monitor safety compliance in industrial and laboratory contexts, associate violations
with individuals, and provide actionable insights through a web-based interface.

## Problem Statement Objective

Design and develop an end-to-end AI-powered system that detects safety compliance in
industrial or laboratory environments using computer vision and machine learning, and
presents results through a working web application.
The system should:
● Analyze live or recorded video feeds from safety-critical environments
● Detect whether individuals are wearing required safety gear:
○ Protective cap / helmet
○ Safety shoes
○ Protective eyewear (specs/goggles)
○ Masks
● Associate detected violations with individuals using face detection, tracking, or
identification techniques.


```
● Log all safety compliance events with accurate timestamps
● Display insights through an admin dashboard
```
## Domain Context

```
● The primary domain for this problem is:
○ Industrial safety (factories, shop floors, plants)
○ Laboratory safety (academic or research labs)
● Teams must choose and clearly specify their target domain (e.g., factory floor or
laboratory).
● The core detection pipeline should remain generic and reusable, even if the demo
focuses on a specific domain.
```
## Functional Requirements

```
● Input:
○ Live webcam feed
○ Pre-recorded video feed simulating an industrial or lab environment
(allowed for demo)
● Computer Vision & ML Requirements:
○ Detect human presence and safety equipment (helmet/cap, shoes,
goggles/specs,mask).
○ Associate the individual with face detection, tracking or identification
techniques.
○ Identify missing safety gear per individual
○ Handle multiple individuals in a single frame (preferred)
● Web Application Requirements(A functional web application that includes):
○ Video feed display (live or recorded)
○ Visual indicators or overlays showing safety compliance status.
○ An admin dashboard displaying:
```
1. Detection logs
2. Compliance / violation statistics
3. Date and time of detection
● Data Storage:
○ The system must store safety compliance records containing:
1. Person identifier (ID or embedding reference)
2. Detected safety equipment
3. Missing safety equipment
4. Timestamp
5. Video or camera source
6. And other required data.

## Expected Deliverables

```
● Working AI System:
○ ML model(s) used for detection
○ Inference pipeline for video input
● Web Application
○ Frontend and backend (any tech stack)
```

```
○ Admin dashboard
● Short solution presentation (template will be provided on the day of hackathon)
along with a live demo.
```
## Constraints & Rules

### Model Usage

```
● Use of pretrained models (e.g., YOLO, SSD, Faster R-CNN, CNNs with transfer
learning) is allowed.
● Using ready-made commercial PPE detection or surveillance SaaS APIs is not
allowed.
● Fully black-box AI solutions without explanation are not allowed.
```
### Development Rules

```
● Participants may use:
○ Publicly available datasets
○ Self-collected datasets
○ A combination of multiple datasets
● A functional web interface demonstrating the solution is mandatory.
○ The application may:
▪ Run locally, or
▪ Be deployed on a cloud platform (optional).
○ Production-grade scalability is not required; focus should be on
correctness and clarity.
● Direct copying of complete solutions or repositories is prohibited.
```
## Evaluation Focus

```
● Machine Learning & Computer Vision Effectiveness
Correctness and robustness of safety gear detection (helmet/cap, shoes, goggles/specs)
under realistic conditions.
● Individual Association Logic
Effectiveness of associating detected safety violations with individuals using face
detection, tracking, or identification techniques.
● System Architecture & Integration
Quality of the end-to-end pipeline, including data flow from video input to detection,
logging, and dashboard visualization.
● Web Interface & Dashboard Clarity
Functionality, clarity, and usability of the web application and admin dashboard for
monitoring compliance events.
```

● Practicality & Real-World Relevance
Feasibility of the solution in industrial or laboratory environments, including handling of
common challenges and stated limitations.
● Innovation & Thoughtfulness
Novel ideas, meaningful enhancements, or thoughtful design choices that improve
robustness, usability, or scalability.