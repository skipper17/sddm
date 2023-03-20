# Common Concern from Reviewers
We thank all the reviewers for their very helpful and constructive comments. We list the common concerns here and post a point-to-point response to each reviewer as well.

### Common Concern 1 (from reviewer HXCZ, YK6e, and gkPe): Need more experiments.

We evaluate our SDDM with other score-based diffusion models on the wild2dog task.

**Rebuttal Table 1.** The comparision between score-based diffusion models in the wild2dog task.
| Model      | FID $\downarrow$| SSIM $\uparrow$|             
| ----------- | ----------- | ----------- |
| ILVR   | 75.33 ± 1.22        | 0.287 ± 0.001|
| SDEdit   | 68.51 ± 0.65        | 0.343 ± 0.001|
| EGSDE (M=0.5T) 1000 diffusion steps     |   59.75 ± 0.62     |     0.343 ± 0.001 |
| SDDM ($T_0$ = 0.5T) 100 diffusion steps   |   57.38 ± 0.53    |  0.328 ± 0.001|

### Common Concern 2 (from gkPe and Kgrj): The extra computation cost of SDDM.

The computational cost for one iteration of the optimization on the manifold of size $C\times H\times W$  with SDDM is $\mathcal{O}(CHW)$ because, in practice, we only iterate at most twice on average at each timestep. 
The computational cost for processing $N$ images of size $C\times H\times W$ with SDDM is $\mathcal{O}(NCHW)$ (including both stages of our proposed method),  which is much smaller when compared with the computational cost of inferring neural networks. 
To better understand this, consider the computational cost of a $3\times 3$ convolution, which is $9CHW$.

### To gkPe

We thank the reviewer gkPe for the valuable comments.
### Q1: Could SDDM capture the distribution of domain $Y$ instead of a single reference image?

Yes. SDDM is capable of capturing the distribution of domain $Y$ using a pretrained energy function or by utilizing the manifold defined by the entire reference dataset. Our implementation has already achieved good results by constructing manifolds of specific image $y_0$, but we can also use the energy function to capture the domain information. In fact, our approach has outperformed both StarGAN v2 and EGSDE. I will provide more details on this in the answer of Q2.

### Q2: Comparison with StarGAN v2, where the FID is much lower than ours.

We sincerely apologize for not providing detailed information about the FID experiment settings in the paper. The FID metric in StarGAN v2 is different from the one used in our submission and CUT. For a fair comparison, we use the public checkpoint of StarGAN v2, which can reproduce the results reported in StarGAN v2 paper to calculate the FID score under the FID metric of CUT in the cat2dog task. In detail, the FID is calculated between 500 generated images and the target val dataset containing 500 images. The results are reported in Rebuttal Table 2, which shows our method outperforms both StarGAN v2 and EGSDE in the cat2dog task.

**Rebuttal Table 2.** The comparision between score-based diffusion models and StarGAN v2 on the popular cat2dog task.
| Model      | FID $\downarrow$| SSIM $\uparrow$|              
| ----------- | ----------- | ----------- |
| StarGAN v2   | 54.87 ± 0.98        | 0.27 ± 0.003|
| EGSDE (M=0.6T) 1000 diffusion steps     |   51.04 ± 0.37     |     0.360 ± 0.001 |
| SDDM ($T_0$ = 0.6T) 100 diffusion steps   |   49.43 ± 0.23    |  0.361 ± 0.001|

### Q3.  Need more diverse comparisons among these models and more visual comparisons.

Thank you very much for the constructive suggestion. We add another wild2dog task in Rebuttal Table 1. We also add the StarGAN v2 in the comparison of the cat2dog task in Rebuttal Table 2.
We will also include more visual results in the final version.

### Q4.  We miss some EBM-based work for baseline comparison.

Thank you very much for the suggestion. We will cite all the papers [1,2,3,4] mentioned and consider including [1] in the comparison datasheet in the final version. We are currently working on this.

###  Q5: It's better to move section 6 into section 2 for a quick description of the Iamge-to-image translation.

Thank you very much for this suggestion. We will combine these parts in the related works section of our paper.

### Q6: Why not fully using the reference statistics can lead to generation being suboptimal?

Referring to Rebuttal Table 3, we observed that the low-pass filter method, compared to BAdaIN, cannot fully utilize the reference statistics, resulting in worse FID and SSIM scores.

**Rebuttal Table 3.** The comparison between the low-pass filter and the BAdaIN, where BAdaIN fully uses the reference statistics. (also Table 2 in our paper)
| Model      | FID $\downarrow$| SSIM $\uparrow$|              
| ----------- | ----------- | ----------- |
| SDDM(low-pass filter)   | 67.56      | 0.411|
| SDDM(BAdaIN)     |   62.29     |     0.422 |

### Q7: How disruption for the intermediate distribution happens and why it can lead to generation being suboptimal?

Firstly, adding a guidance function to the SDE creates a new SDE that cannot avoid perturbations in the original immediate distribution. Secondly, the linear combination of the guidance functions and score function has the probability of negative impact (PNI) from guidance functions to score function, and the rebuttal table 4 shows the comparison.

**Rebuttal Table 4.** The comparison between linear combination and multi-objective optimization of the guidance functions and score function.
| Model      | FID $\downarrow$| SSIM $\uparrow$| PNI $\downarrow$|      
| - | - | - |- |
| SDDM(linear combination)   | 64.93      | 0.421| 0.024 |
| SDDM(multi-objective optimization)     |   62.29     |     0.422 | 0 | 


###  Q8:  How does it compare with models based on perceptual supervision?

We compare our SDDM with StarGAN v2 based on perceptual supervision in the Rebuttal Table 2.
Our framework is also compatible with various perceptual methods, including the energy function based on perceptual supervision in EGSDE, which can enhance the quality of our results.


[1] Learning Cycle-Consistent Cooperative Networks via Alternating MCMC Teaching for Unsupervised Cross-Domain Translation. AAAI 2021.

[2] Cooperative Training of Descriptor and Generator Networks. PAMI 2018

[3] Learning Energy-Based Generative Models via Coarse-to-Fine Expanding and Sampling. ICLR 2021. 

[4] A theory of generative ConvNet. Icml 2016.

### To HXCZ

We thank the reviewer HXCZ for the acknowledgment of our contributions very much.
### Q1: Experiments in Table 1 show that the improvement of the proposed method against EGSDE is not significant enough.

Thank you very much for your valuable comments. The energy function used in EGSDE is strongly pretrained on related datasets, and thus contains significant domain-specific information. To demonstrate the effectiveness and versatility of our framework, we intentionally chose to use a weak energy function, consisting of only one layer of convolution, without any further pretraining. Despite this, our approach consistently outperformed EGSDE and other SOTA methods on various image-to-image translation tasks.

### Q2:  There are only quantitative comparisons in the paper. It will be better if some visual comparisons can be added.

Thank you very much for the valuable suggestions. We apologize for not including enough visual comparisons in our paper. We will include more visual results in the final version in addition to Figure 4. to address this concern.

### To YK6e
We thank reviewer YK6e for the valuable comments.

### Q1: The performance with sufficient diffusion steps is not as good as the state of the art.

We apologize that the table may have caused some confusion, leading to a misunderstanding. Specifically, in the Cat2Dog task, our SDDM model with 100 diffusion steps outperforms EGSDE with 1000 diffusion steps, while in the male2female task, our SDDM model with 100 diffusion steps outperforms EGSDE with 200 diffusion steps. 

While it is true that EGSDE with 1000 diffusion steps outperforms our 100 diffusion steps SDDM, it is important to note that the energy function used in EGSDE is strongly pretrained on related datasets and contains significant domain-specific information. In contrast, to demonstrate the effectiveness and versatility of our framework, we intentionally chose to use a weak energy function consisting of only one layer of convolution without any further pretraining. After incorporating the strong guidance function from EGSDE, our method outperforms EGSDE in the FID score, as shown in the Rebuttal Table 5. We will clarify this point in the final version of the paper.

**Rebuttal Table 5.** The FID comparison between EGSDE and our SDDM with the same energy guidance function.
| Model      | FID $\downarrow$ |           
| ----------- | ----------- |
| EGSDE (M=0.5T) 1000 diffusion steps    | 41.93 ± 0.11        |
| SDDM ($T_0$ = 0.5T) 100 diffusion steps  |   40.08 ± 0.13    | 


### Q2: Any theoretical justification for what manifolds are more suitable in practice?

Thank you for the constructive question. 
There is an analytics-based approach by measuring the distance between distributions. Since the manifold in SDDM represents the manifold of the perturbed $y_0$, we can calculate the KL distance between the distribution indicated by the manifold of perturbed $y_0$ and the distribution indicated by the real data manifold (like in the second part of FID calculation) to determine whether the manifold is tight. And the tightness of the manifold can somehow indicate whether the manifold is suitable.

In practice, we use hyperball restriction, and the key challenge is how to choose the number of chunking blocks. Within a certain range, chunking more blocks means a lower dimension of the manifold, which leads to tighter constraints and better SSIM scores but worse FID scores. We chose a block num of 16x16, considering both the visual result and the balance between FID and SSIM scores as indicated in Rebuttal Table 6.

**Rebuttal Table 6.** The comparison of different block num settings.
| Model  | block num    | FID $\downarrow$ |   SSIM $\uparrow$|     
| - | - | - | -|
| SDDM     | 8x8    |54.56 ± 0.93  | 0.359 ± 0.002 |
| SDDM     | 16x16    | 62.29 ± 0.63 | 0.422 ± 0.001|
| SDDM     | 32x32    | 68.03 ± 0.47  | 0.426 ± 0.001 |

### Q3: Lack of discussions about the limitations of the method.

Thank you very much for this valuable comment. Our method will introduce extra computation costs, as discussed in Common Concern 2. We will add the discussion in our final version.

### To Kgrj
We thank reviewer Kgrj for the valuable and constructive comments.

### Q1: The novelty compared with DVCE[5].

- Previous methods, such as EGSDE, use fixed coefficients for the gradients of guidance functions and score during the entire diffusion process.
- DVCE takes a step forward by first normalizing the gradients of guidance functions first, but still uses fixed coefficients after normalization during the entire diffusion process.
- While previous methods, such as EGSDE and DVCE, have focused on balancing the guidance functions, **they have neglected to consider the balance between the guidance functions and the score.** They both can not avoid the case that the guided score may have a negative direction with the original score, which indicates the guidance functions have an overly negative impact on the score. 
- In contrast, our approach utilizes multi-objective optimization to address this issue. By dynamically changing the coefficients of the gradients of guidance functions and score at different diffusion steps, we can ensure that negative impacts are avoided. You can refer the Table 4. for details.
In summary, our framework takes into account not only balancing the guidance functions but also balancing the guidance functions and score.
I missed this paper before, but I will make sure to include it in the final version of my work and properly cite it.

### Q2: The novelty compared with  Improving Diffusion Models for Inverse Problems using Manifold Constraints[6].
This method has a strict linear form of the condition, which is an overly strong assumption for many conditional generation tasks like unpaired image-to-image translation. Additionally, it relies on the strong manifold assumption for theoretical results. In contrast, our method follows manifold optimization at each time step of the stochastic differential equation. We use the restriction function as a bridge between the manifold and the tangent space and establish the relationship between the AdaIN module (which is generally used in style transfer) and the restriction function. Furthermore, we propose the BAdaIN module for better manifold design. Our proposed method can be applied more effectively to a wider range of conditional generation tasks including the unpaired image-to-image translation.
It is also worth noting that in our concentration of perturbed manifold, we reduce the dimensionality bound of the manifold from n-1 to n-2, and further reduce it using chunking techniques.

### Q2: The novelty compared with IDMIPMC[6].
This method has a strict linear form of the condition, which is an overly strong assumption for many conditional generation tasks like unpaired image-to-image translation. Additionally, it relies on the strong manifold assumption for theoretical results. In contrast, our method follows manifold optimization at each time step of the stochastic differential equation. We use the restriction function as a bridge between the manifold and the tangent space and establish the relationship between the AdaIN module (which is generally used in style transfer) and the restriction function. Furthermore, we propose the BAdaIN module for better manifold design. Our proposed method can be applied more effectively to a wider range of conditional generation tasks including the unpaired image-to-image translation.
It is also worth noting that in our concentration of perturbed manifold, we reduce the dimensionality bound of the manifold from n-1 to n-2, and further reduce it using chunking techniques.

### Q3: The novelty compared with the predictor-corrector method in score-SDE[7].

While the tick-tock formulation may appear similar between our method and the predictor-corrector method, they are actually two distinct methods with different objectives. The predictor-corrector method is intended for stable generation, whereas our method is designed to incorporate conditional information and balance the score and guidance functions.

Take the time step t+1 to t as an example:
The tick steps, which refer to the predictor of Score-SDE and the transformation between adjacent manifolds in our method, are both aimed at denoising. However, our method uses a manifold that is carefully designed to seamlessly inject conditional information.
The tock steps, which refer to the corrector of Score-SDE and the optimization on the manifolds in our method, have different purposes. The corrector of Score-SDE is using the MCMC approach to ensure the image obeys the distribution of the t-level noise perturbed data, while our tock step is proposed to reach the local balance point of score and energy guidance, and also in order to avoid the energy guidance having an overly negative effect on the score.

We will add the discussions of Q1-3 briefly in our final version.

### Q4:  The dynamic balancing of the multiple functions for guidance is not verified empirically.

Thank you very much for this constructive comment.
The primary objective of the multi-objective optimization in our method is to achieve a dynamic balance between the guidance functions and the score. Details are shown in Table 4. 

It is worth noting that our proposed method can also dynamically balance multiple guidance functions.

We incorporated two guidance functions from EGSDE into our method and evaluated the SDDM's ability to dynamically balance multiple guidance functions. We present the results for the male2female task in the following table, where we did not use any guidance functions when $t < T_0$ for a fair comparison. The results demonstrate that our proposed method can effectively balance multiple guidance functions.
**Rebuttal Table 7.** The comparison of different balance methods.
| Model      | FID $\downarrow$ |  SSIM  $\uparrow$|          
| ----------- | ----------- | -----|
| SDDM ($T_0$ = 0.5T) w/ the simple guidance function    | 44.37 ± 0.23        | 0.526 ± 0.001 |
| SDDM ($T_0$ = 0.5T) w/ another two guidance functions w/o MOO  |   44.23 ± 045    | 0.532 ± 0.001|
| SDDM ($T_0$ = 0.5T) w/ another two guidance functions w/ MOO |   42.24 ± 0.31    | 0.535 ± 0.001|  

### Q5:  Whether need to project of $x_t^*$ onto B at the last line of Algorithm 1.

Thank you very much for the valuable comment. For the compactness of our proposed algorithm, it is necessary to project $x_t^*$ onto B. In practice, we fix the number of iteration steps for minimal extra computational cost and get comparable results.

### Q6: What is the meaning of "control the impact of the reference image on the generation process"?

We apologize for not making it clear. It means by controlling the chunking blocks (changing the manifold), we can adjust the influence of the reference image. We will explain it further in our final version.

### Q7:  Why the setting of $\lambda_1$ affects the performance of the proposed method in Table 3? 

I apologize for not providing a clear explanation earlier. The $\lambda_1$ is actually the coefficient after the normalization step. In other words, we first normalize the guidance functions to have the same norm as the score and then assign the guidance coefficients. Our coefficients are designed according to the following features of multi-objective optimization.
- vectors with larger norms have a smaller impact on the final sum vector.
- the final sum vector has a smaller norm than all the vectors.
For more information, please refer to the paper "Multiple-gradient descent algorithm (MGDA) for multiobjective optimization".


### Q8: When we use many functions simultaneously for guidance, some of them might be ignored due to the multi-objective optimization, because any beta can be zero after the optimization.

Thank you very much for the valuable comment. At low probabilities, some guidance functions may have an ignorable norm but **only at specific timesteps where it would have an overly negative impact on the score function, and avoiding the overly negative impact on the score function is what we want.** In the next timestep, when it does not negatively impact the score function, its norm will not be ignorable anymore. Therefore, the ignorable norm is not permanent and only exists temporarily at certain timesteps.

### P.S.

We apologize that there exists several typos and improper citations. We will carefully check them in the final version.

[5] Diffusion Visual Counterfactual Explanations. NeurIPS 2022.

[6] Improving Diffusion Models for Inverse Problems using Manifold Constraints. NeurIPS 2022.

[7] Score-based Generative Modeling through Stochastic Differential Equations. ICLR 2021.