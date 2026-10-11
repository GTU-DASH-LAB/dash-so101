# Social copy — Instagram caption + LinkedIn post

Write both after the reel is exported. Give each in its own ```text block so it pastes cleanly.
Facts in the copy must match the video and `dataset_assets.py stats` (no invented numbers).

## Voice rules (from the user's feedback on Day 1)

- **Sound like a human, not a template.** First person, short sentences, a little story. No arrow lists,
  no "The twist:", no numbered "Why?" sections, few emojis on LinkedIn.
- **Admit uncertainty** when results aren't in yet ("I honestly don't know if it'll work") — it makes the
  next episode the payoff.
- **Talk about the series**: it's a new series, one day at a time, until someone notices. Day N, Day N+1…
- **Thanks = ONE short line.** "Special thanks to Gebze Technical University and GTU DASH Lab for making this
  possible. 🙏" — the user explicitly said: speak less but speak well.
- Don't repeat the MSc line if it adds length — the LinkedIn profile already shows it.
- Point to the next day, and end with one genuine question to invite comments.
- Hashtags: Instagram max **5**; LinkedIn **3**.
- If the training is still running when posting, write "training", not "trained".
- Dataset name in copy must match the published HF repo (check before posting).

## Profile facts

- Fouad Aladhami — MSc in AI (Computer Engineering), Gebze Technical University, GTU DASH Lab
- Instagram: **@engineer__fouad**
- LinkedIn: **linkedin.com/in/fouad-aladhami** (show as `in/fouad-aladhami`)
- GitHub: **github.com/fouad1233**; lab: **github.com/GTU-DASH-LAB**
- Hugging Face: **fouad1233** (Day-1 dataset `fouad1233/so101_pen_in_bowl_wrist`)

## Instagram caption template (Day 1 as delivered)

First line = the hook shown before "more".

```text
Day 1 of making robotics videos until I get recognized 🤖

Today I taught a robot arm to pick up a pen and drop it in a bowl, using only ONE camera.

What we did:
🎮 Teleoperated the SO-101 arm with a leader arm
📼 Recorded 100 demos (65,550 frames)
📷 Used one camera, mounted on the gripper
🧠 Trained an ACT policy (Action Chunking Transformer) on 2 GPUs

The idea: for a simple task, one wrist camera should be enough. The camera moves with the gripper, so the robot sees the same view wherever it is. It might generalize better than a fixed overhead camera.

Will it work? Real-robot results in Part 2 👀
Follow so you don't miss Day 2 🔔

💻 GitHub: fouad1233
🧪 Lab: GTU-DASH-LAB @ Gebze Technical University
💼 LinkedIn: in/fouad-aladhami

👇 What task should the robot learn next? Tell me in the comments.

#robotics #ai #machinelearning #lerobot #imitationlearning
```

For Day N: swap the hook line ("Day N of …"), the one-sentence "today I …", the 3–4 "what we did" lines,
the idea/result paragraph, "Follow so you don't miss Day N+1". If Day N delivers results, lead with them
("It worked 7 out of 10 times. Here's what failed.").

## LinkedIn post template (Day 1, final approved version)

```text
Today I took a camera away from my robot.

But first, some news: this is Day 1 of a new series. I'm going to share what I build in robotics, one day at a time, until someone notices. Real experiments, real failures, and hopefully a few things that actually work. Day 1, Day 2, Day 3... let's see how far it goes.

For the first episode I'm working with an SO-101 arm. The task is simple on paper: pick up a pen and put it in a bowl. I controlled the arm by hand with a second "leader" arm and recorded 100 demonstrations, about 65k frames. Then I trained an ACT policy on it on two RTX 5090s.

In my earlier experiments I always used two cameras: one looking down at the table and one on the gripper. This time I kept only the gripper camera.

My thinking: the robot doesn't need to see the whole room to grab a pen. A camera on the gripper sees roughly the same thing no matter where the pen is, and that might help it generalize. It also makes the setup cheaper and simpler.

But I honestly don't know if it'll work. Without the overhead view, the robot has much less idea of where it is in the scene. It might just wander around looking for the bowl.

I'll find out on the real robot and post the results in Day 2.

The dataset is public on Hugging Face (fouad1233/so101_pen_in_bowl_wrist) if anyone wants to play with it.

Special thanks to Gebze Technical University and GTU DASH Lab for making this possible. 🙏

GitHub: github.com/fouad1233
Lab: github.com/GTU-DASH-LAB

If you want to follow the series, hit follow so you catch Day 2. And if you've worked with wrist-only cameras before, did it hold up for you? Genuinely curious.

#Robotics #ImitationLearning #LeRobot
```

Structure to keep for Day N: a one-line hook that is a small surprising statement → (Day N>1) one line
reminding where the series is ("Day N of the series…") → setup in plain words → what's different today →
my reasoning → honest uncertainty or the actual result → what's next → dataset/model link → ONE-line thanks →
links → follow + one genuine question → 3 hashtags.

Posting tips to mention briefly: links in the body can reduce LinkedIn reach (option: move them to the first
comment); tag the Gebze Technical University page and GTU DASH Lab in the thanks line.
