---
layout: post
title: The Enlightened Hunt-and-Pecker
description: "It was day one of programming class: enter Vim, the commend-line, C code, and last but not least, hunt-and-peck typing."
modified: 2016-01-05
tags: [Data Visualization, Psychology, Learning]
image:
  feature: bum-master.jpeg
---

It was day one of programming class: enter Vim, the commend-line, C code, and last but not least, hunt-and-peck typing. It was a recipe for disaster — inspiring none other than this dystopian parody of Eminem’s Lose Yourself:

>Look, if you had, one shot, or one [lab]
To seize [all the points] you ever wanted. In one moment
Would you capture it, or just let it slip?
Yo
His palms are sweaty, fingers weak, [recursion is] heavy
There’s vomit on his [pocket protector] already, [code’s] spaghetti
He’s nervous, but on the surface he looks calm and ready to drop [semicolons],
But he keeps on forgetting what he wrote down,
The whole [keyboards of other students] go so loud
He opens [the editor], but the words won’t [type] out
He’s choking how, everybody’s [submitting] now
The clock’s run out, time’s up, over, blaow!

And it was at that moment during the first day of lab that I realized being the owner of two bundles of floppy noodle fingers wouldn’t do. Instead, I envisioned myself as an industrious German secretary — effortlessly hammering out a euphony of text at 100 words per minute.

After reading the title, you might have got the hint that this vision got a bit out of hand, and you would be right. So am I sitting at this keyboard now, pushing Medium’s sleek typing interface to the brink of failure with my blazing fast typing speed, or are my hands still moving like a pair of frost-bitten nubs glazed in molasses? Stay tuned to find out.

# Hour 1: Welcome to Boot Camp
The first step towards becoming an enlightened hunt-and-pecker was picking the best typing program to use. Despite the appeal of the kid tested mother approved typing games like “trick or type” or “spacebar invaders”, I decided to opt for the nerdiest one I could find — [Keybr.com](http://www.keybr.com/).

Coining itself as the best way to “learn typing at the speed of thought”, Keybr has some pimped out features. The interactive dashboard presents new letters one at a time, forcing “mastery” of the previous ones before new are presented. It even breaks down response time by letter each sample set.

The words (or pseudo-words) are also constructed based off of the natural statistical patterns in common text, encouraging muscle memory of common finger movements. Much to my dismay, the interface continues to present the letter lowest in accuracy until it gets back up to par with the others.
<figure>
   <a><img src="{{site.url}}/assets/img/ehp2.png" alt=""></a>
      <figcaption>Typing interface used to practice and provide feedback.</figcaption>
</figure>

I’ll dissect the interface more below and talk about how it benefited (or didn’t) the learning process. For the most part, the free software made for a homey training ground — a paradise industrious German secretaries only dream of.

# Queue the Montage Music

Once I was all situated, it was time for a typing montage. Far from an aspiring aficionado ruthlessly training away for hours at a time, I usually just typed for a few minutes a day. It became a favorite procrastination tool, and may be more addicting than Flappy Bird.

After a few hours of typing practice, I decided to make my 100 word-per-minute goal more realistic, reducing it to simply teasing all the letters out of Keybr’s mastery-oriented interface. Although I imagined the process being quick and simple, it was more tricky.
<figure>
   <a><img src="{{site.url}}/assets/img/ehp4.png" alt=""></a>
      <figcaption>Smoothed scatter plot of typing speed and accuracy — provided by Keybr.</figcaption>
</figure>
As you can see, learning was more like trudging up a mountain than taking an elevator — especially because learning how to type required abandoning my hunt-and-peck ways. When it came time to write a paper or program, I had to revert back to chugging along until I learned most of the letters the correct way,

As the hours drifted past and my motivation waned, I began to wonder: is the process of learning typing fixed, or is there a way to increase typing speed more quickly? This lead to some exploration of the Keybr page.

# There’s Data?!

Around twenty hours into the voyage, I stumbled into a trove of gold — a complete record of the data used to change the letter of focus and display progress. This meant that, in addition to the pretty graphics provided by Keybr, I could generate my own to explore the learning process. I was particularly interested in the reason some letters quickly became etched into muscle memory, while others drug on for ages.

Although the data portal didn’t provide letter correlations or a miss-hit-ratio, it was easy to derive with the downloaded data using Python. The code used to convert the raw data to JSON and generate the plots is on this [Github repository](https://github.com/lguerdan/type-training), feel free to take a gander if you’re feeling curious. Using the code, I was able to generate a correlation matrix which breaks down the comparison between response time of each letter. The closer to red, the more the two letters match in average response time across the training.
<figure>
   <a><img src="{{site.url}}/assets/img/ehp5.png" alt=""></a>
   <figcaption><a title="Correlation matrix of low response time between letters."> Correlation matrix of low response time between letters.</a></figcaption>
</figure>

### Some interesting observations:

- Letters matching the same finger (left verses right) interfered
- Letters matching the same finger (on the same hand) interfered
- Vowels were more likely to interfere with other vowels
- Consonants were more likely to interfere with other consonants
- Performance on some letters decreased over time :(

When letters matched multiple situations, getting a snappy response time was especially tricky, requiring a large amount of effort to master the associated letters. For example, the M-F duo featured two consents with the same finger on opposite hands. The K-X combination was the same.

As I slowed down to examine why this was occurring, I noticed that although I attempted to hit M, the letter F was triggered by my finger on accident. This nightmarish situation was like playing alphabet whack-a-mole, with success in one letter spelling demise for the other. Yet, when I slowed down my pace by around 20 words-per-minute, it became much easier to control which finger went where.

After discovering these clues (and some googling), I finally found the reason increasing typing speed took so long.

# Hour 26: Typing Enlightenment
The reason it took so long to reach typing proficiency, it turns out, is the phenomenon of [*retroactive interference*](https://en.wikipedia.org/wiki/Interference_theory). In this particularly mischievous form of interference, newly acquired information inhibits our ability to recall previously acquired information — in other words, as I learned new letters, I continued to develop issues recalling older ones.

In this two steps forward one step back approach, I would become proficient enough to move on to the next letter, while still being poor enough to have the information wiped away later on. This meant that despite the deceiving rate of new letter learning, the accuracy improved much slower than it could have.

What made this issue so difficult to detect was that I assumed *proactive interference* was the culprit. The converse of it’s retroactive sister, proactive interference is the tendency of previously learned material to hinder later learning — the letters I already knew were simply slowing my progress with new ones. *Why wasn’t I getting more proficient with letters?* I assumed it was because of this!

The cure to this form of interference — it turns out — is to focus on mastery over speed. Once a sufficient level of mastery is reached, interference from new letters decreases and the previously learned letters remain stable. Once I learned this at the end of boot camp, my proficiency increased much more rapidly and the alphabet whack-a-mole was finally decommissioned. To address this, Keybr could simply adjust it’s settings to require more time with previously learned letters when performance decreases.

## After over a day of pounding away and dash of enlightenment, what are the takeaways?

1. **Mastery over speed.** When I first sat down to begin typing, I was too focused on becoming a speed machine, and not enough on becoming a typing sniper. When accuracy is honed from the get-go, interference is reduced and the phalange flailing is kept to a minimum.
2. **Montages over marathons.** One long period of typing is much less beneficial than the same amount of time broken over a series of days. Although I didn’t completely fail in this regard, the two-hour blocks of practice often resulted in mindless typing which ignored focused improvement.
3. **Learn unlearning.** The process of retroactive interference meant walking through quicksand when a paved path was a few feet away. Although the solution to this is the first point, I could have overcome the issue much easier by understanding how it works earlier.

What’s best is that these discoveries don’t simply apply to typing lessons— they also fit well with other procedural learning activities such as instruments, sports, and even some aspects of programming. Although the learning process is rewarding, what’s better yet is learning how to learn. Had I understood what I do now, what took 26 hours could easily be compressed into 13, leaving twice the time for writing stellar Eminem parodies.

Although I’m by no means a certified typist, I’m happy to have overcome hunt-and-peck syndrome and moved to a respectable 55 words per minute (not quite enough to break Medium’s typing interface). Nonetheless, dissecting the learning process was a blast. Hopefully my little sliver of enlightenment inspires many learning adventures to come.
Thanks for reading!

