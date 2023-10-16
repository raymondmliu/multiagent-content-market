# Overview

In this model, we consider a community on Twitter to be comprised of three types of agents: content consumers, content producers, and a central influencer who shares content and provides information on the utility of the content shared. Note that content producers and consumers can be the same individuals in the community, but we consider them to be different agents due to their differing goals and roles.

These members of the community consume, produce, and share content on specific topics, and have limited resources to do so. The content producers and influencers have the goal of maximizing the amount of social support (i.e. the amount of likes, retweets, comments on Twitter) they receive for their content. The content consumers obtain content that is of interest to them, and can showcase their interest by indicating social support for the content.

This creates a content market in which social support acts as a currency. Existing work has been done on proving the existence of this content market and identifying certain characteristics using game-theoretic mathematical models \[link paper here\], as well as applying the idea of this core-periphery structure to analyze Twitter data \[link Twitter analysis here\]. In this notebook, we extend this existing research by implementing an agent-based model, using the mathematical formalisms provided in the paper. Each agent will be assigned a utility function based on their specific goals, and their actions will be modelled as optimizing a utility function.

[TODO: add a general overview on the process for optimizing the utility function, once that has been implemented.]

# Setup

First, note that we represent the set of topics $T$ as an interval. For this model, let us set

$$T = (0, 1) \subseteq R.$$

Note that we consider the closeness between any two topics $x, y \in T$ to be the distance $|x-y|$.

Furthermore, for each community member, we identify them by a main interest $y \in T$.

The probability that a community member $y \in T$ is interested in consuming content on topic $x \in T$ is:
$$p(x|y) = f(|x-y|)$$

where $f : \mathbb{R}_+ \to [0,1]$ is strictly decreasing and continuous on the support of $f$. For now, we set it as
$$f(x) = 1-x.$$

The ability that a community member $y \in T$ has to produce content on topic $x \in T$ is:
$$q(x|y) = g(|x-y|)$$
where $g : \mathbb{R}_+ \to [0,1]$ is strictly decreasing and continuous on the support of $g$. For now, we set it as
$$g(x) = 1-x.$$