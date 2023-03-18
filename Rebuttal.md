## To gkPe

**W1:**  <1> *In section 2.2, according to the author's definition of image-to-image tasks, it looks like the proposed model is conditioned on a specific image $y_0$. Could SDDM capture the distribution of domain Y instead of a single reference image?*

Yes, SDDM is capable of capturing the distribution of Domain Y using a pretrained Energy function or by utilizing the manifold defined by the entire reference dataset. Our implementation has already achieved excellent results by constructing manifolds of specific image $y_0$, but we can also use energy function to capture the domain information. In fact, our approach has outperformed both StarGAN v2 and EGSDE. I will provide more details on this in W2.


<2>*What is the computation cost for the transfer of given N images in the source domain?*

The computational cost for processing N images of size CxHxW with SDDM is O(NCHW), which is negligible when compared to the computational cost of inferencing neural networks.
To better understand this, let's consider the computational cost of a 3x3 convolution, which is 9CHW.

**W2:** *In 5.1, the GAN models that the author list are pretty old. The GAN model could achieve 13.7 in CelebA-HQ and 16.2 in AFHQ in [1], 7.79 in CelebA-HQ, and 8.04 in FFHQ. It is not fair to compare only with old models.*

We sincerely apologize for not providing detailed information about the FID experiment settings in the paper. Since the diffusion process is slower than GANs, we followed EGSDE's approach and reduced the number of generated images for the FID comparison. We will also include the results of StarGAN v2 for a fair comparision. In the Cat2Dog task, our method (with strong guidance function of EGSDE) outperformed both StarGAN v2 and EGSDE, as shown in the following table. We will also include the FID results of other datasets in the final version.
| Model      | FID | SSIM|             
| ----------- | ----------- | ----------- |
| StarGAN v2   | 54.88 ± 1.01        | 0.27 ± 0.003|
| EGSDE (M=0.6T) 1000 diffusion steps     |   51.04 ± 0.37     |     0.361 ± 0.001 |
| SDDM ($T_0$ = 0.6T) 100 diffusion steps   |   49.43 ± 0.23    |  0.361 ± 0.001|

**W3:** *In table 1, 62.29 and 42.37 are not impressive performance on these two datasets. I expect to see more diverse comparisons among these models. (The cases shown in fig4 and fig5 are too limited).*

See the W2 part, we are sorry for not explaining the experiment settings clear again. Our model achieved state-of-the-art FID scores in these tasks. We will also include more visual results in the final version.

**W4:** *Moreover, the paper currently misses the following EBM-based work for baseline comparison, such as [3] uses Cooperative learning， [4] for unpaired image-to-image translation， [5] uses an energy-based framework, [6] with a short-run MCMC (i.e., Langevin dynamics) for unpaired image-to-image translation.* 

We will cite all the papers mentioned and consider including [3] in the comparison datasheet in the final version. We are currently working on this.

**W5:** *It's better to move section 6 into section 2 for a quick description of the Iamge-to-image translation.*

Thank you for your feedback! We will combine these parts in the related works section of our paper.

**Q1:** *In section 2.2, the author claims that the generation could be suboptimal as it cannot fully use the reference statistics and can not avoid disruption for the intermediate distribution. Could the author show any evidence and papers to support this argument?*

<1> Referring to Table 2 in our paper, we observed that the low-pass filter method, when compared to BAdaIN, cannot fully utilize the reference statistics, resulting in worse FID and SSIM scores.

<2> Firstly, adding a guidance function to the SDE creates a new SDE that cannot avoid perturbations in the original immediate distribution. Secondly, the linear combination of the guidance function and score function can have a negative impact on the score function, as shown in the PNI in Table 4 of our paper. This is one of the reasons why we introduce multi-objective optimization.


**Q2:**  *Since this is an image optimization-based model, how does it compare with models based on perceptual supervision?*

Our framework is compatible with various perceptual methods, including the energy function based on perceptual supervision in EGSDE, which can enhance the quality of our results. Additionally, we will compare our approach with StarGAN v2, as outlined in W2.

**L1:** *Please refer to the weakness. Although the author proposes an interesting idea to achieve translation by diffusion models, the performance looks far away from the recent state-of-the-art models on the benchmark. And the work also misses comparison with EBM models for I2I translation.*

The performance has already been explained in the W2 section, and we plan to include EBM methods in the final version.

## To HXCZ

**W1:** *Sec 2.2 (Line 109) says that EGSDE may achieve only sub-optimal results, but experiments in Table 1 show that the improvement of the proposed method against EGSDE is not significant enough. And the comparisons with other methods only include two specific tasks. It will be better if more general comparisons can be added.* 

Thank you for your insightful questions. The energy function used in EGSDE is strongly pretrained on related datasets, and thus contains significant domain-specific information. To demonstrate the effectiveness and versatility of our framework, we intentionally chose to use a weak energy function, consisting of only one layer of convolution, without any further pretraining. Despite this, our approach consistently outperformed EGSDE and other SOTA methods on various image-to-image translation tasks.
Moreover, we acknowledge the importance of evaluating our approach on more diverse and challenging tasks, such as wild2dog image translation, and we consider including such comparisons in the final version of our paper. 

[ To dispel your doubts, we introduce the strong energy function of EGSDE in our method and get the following result. the SSIM of EGSDE is better because EGSDE and SDEdit use noisy reference image as input directly.]

**W2:**  *There are only quantitative comparisons in the paper. It will be better if some visual comparisons can be added.*

Thank you for bringing up this point. We apologize for not including enough visual comparisons in our paper. While some visual comparisons are presented in Fig. 4, we recognize that more examples would help better illustrate the strengths and limitations of our method. We will make sure to include additional visual results in the final version of the paper to address this concern.

## To YK6e

**W1:** *The experiments are only based on two datasets, which is limited. The performance with sufficient diffusion steps is not as good as state of the art.*

<1> We acknowledge the importance of evaluating our approach on more diverse and challenging tasks, such as wild2dog image translation, and we consider including such comparisons in the final version of our paper. 

<2> The table may have caused some confusion, leading to a misunderstanding. Specifically, in the Cat2Dog task, our SDDM model with 100 diffusion steps outperforms EGSDE with 1000 diffusion steps, while in the male2female task, our SDDM model with 100 diffusion steps outperforms EGSDE with 200 diffusion steps. While it is true that EGSDE with 1000 diffusion steps outperforms our 100 diffusion steps SDDM, it is important to note that the energy function used in EGSDE is strongly pretrained on related datasets and contains significant domain-specific information. In contrast, to demonstrate the effectiveness and versatility of our framework, we intentionally chose to use a weak energy function consisting of only one layer of convolution without any further pretraining. After incorporating the strong guidance function from EGSDE, our method outperforms EGSDE in the FID score, as shown in the following table. We will clarify this point in the final version of the paper.
| Model      | FID |           
| ----------- | ----------- |
| EGSDE (M=0.5T) 1000 diffusion steps    | 41.93 ± 0.11        |
| SDDM ($T_0$ = 0.5T) 100 diffusion steps  |   40.08 ± 0.13    | 


**Q1:** *Any theoretical justification what manifolds are more suitable in practice?*

Thank you for the interesting question. In practice, we use hyperball restriction, and the key challenge is how to choose the number of chunking blocks. Within a certain range, chunking more blocks means a lower dimension of the manifold, which leads to tighter constraints and better SSIM scores but worse FID scores. We chose a block size of 16x16, considering both the visual result and the balance between FID and SSIM scores.

[TODO] 贴不同block个数的对比

There is also an analytics-based approach. Since the manifold in SDDM represents the manifold of the perturbed $y_0$, we can calculate the KL distance between the distribution indicated by the manifold of perturbed $y_0$ and the distribution indicated by the real data manifold (like in the second part of FID calculation) to determine whether the manifold is tight. We will include this analysis in the final version of the paper. Thank you for your valuable question.


L1: *The paper should discuss more about the limitations of the method.*  

-   Our method requires additional computations for manifold construction and multi-objective optimization compared to traditional methods.
-   Due to the manifold is inappropriate for the low-noise images, it can not be applied in the final 10% diffusion steps.

## To Kgrj

**W1:** *Several related studies are missed. Due to this, I could not sufficiently judge the novelty as well as significance of this work.*

We introduce a novel framework for conditional image generation through diffusion models that does not require pre-training diffusion model with conditions. Our approach is particularly focused on image conditioning.

<1> Compared with Diffusion Visual Counterfactual Explanations (DVCE)
- Previous methods, such as EGSDE, use fixed coefficients for the gradients of guidance functions and score during the entire diffusion process.
- DVCE takes a step forward by first normalizing the gradients of guidance functions first, but still use fixed coefficients after normalization during the entire diffusion process.
- While previous methods, such as EGSDE and DVCE, have focused on balancing the guidance functions, they have neglected to consider the balance between the guidance functions and the score. They both can not avoid the case that the guided score may have negative direction with original score, which indicades the guidance functions have overly negative impact on score. 
- In contrast, our approach utilizes multi-objective optimization to address this issue. By dynamically changing the coefficients of the gradients of guidance functions and score at different diffusion steps, we can ensure that negative impacts are avoided. You can refer the Table 4. for details.
As a summary, our framework takes into account not only balancing the guidance functions, but also balancing the guidance functions and score.
I missed this paper before, but I will make sure to include it in the final version of my work and properly cite it. Thank you very much!

<2> Compared with Improving Diffusion Models for Inverse Problems using Manifold Constraints
This method has a strict linear form of the condition, which is a overly strong assumption for many conditional generation tasks like unpaired image-to-image translation. Additionally, it relies on the strong manifold assumption for theoretical results. In contrast, our method follows manifold optimization at each time step of the stochastic differential equation. We use the restriction function as a bridge between the manifold and the tangent space, and establish the relationship between the AdaIN module (which is generally used in style transfer) and the restriction function. Furthermore, we propose the BAdaIN module for better manifold design. Our proposed method can be applied more effectively to a wider range of conditional generation tasks including the unpaired image-to-image translation.
It is also worth noting that in our concentration of perturbed manifold, we reduce the dimensionality bound of the manifold from n-1 to n-2, and further lower it using chunking techniques.

<3> Compared with Score-SDE
The predictor-corrector method is amazing, but we are totally different. from t to t-1, predictor and corrector are both to keep the distribution of t-1
But in our method we try to reach the local balance point of score and energy guidances.

**W2:**  *Although one of the important advantages of the proposed method is the dynamic balancing of the multiple functions for guidance, this point has not been verified empirically, because the authors used only a single energy function for guidance in the experiments.*

Sorry for not make it clear that the main purpose of the multi objective optimization is dynamaic balancing the guidance functions and the score, more details can refer the W1.1 and it is verified in  Table 4. 
Indeed our proposed method also can dynamic balancing multiple guidance functions. 
We add the two guidance function from EGSDE and here is the balance results for the male2female task (For a fair comparision we do not use guidance functions when $t < T_0$ ), which shows that our proposed method can also dynamic balance multiple guidance functions.
| Model      | FID |  SSIM |          
| ----------- | ----------- | -----|
| SDDM ($T_0$ = 0.5T) w. the simple guidance funciton    | 44.37± 0.23        | 0.526 ± 0.001 |
| SDDM ($T_0$ = 0.5T) w. another two guidance functions  |   42.24± 0.31    | 0.535 ± 0.001|  which indicades our method works

**W3:**  *At line 13 of Algorithm 1, $x_t$ should be $x_t^*$. I'm not sure whether we need projection of $x_t^*$ onto B after this line or not.*

typo, according to the xxx, we can save this restriction step.

**W4:** *It is not clear to see what the authors mean by "control the impact of the reference image on the generation process" at line 285.*

It means we can control the chunking blocks (change manifolds) to change the impact of reference image.

**W5:**  *I could not understand why the setting of $\lambda_1$ affects the performance of the proposed method as shown in Table 3. The gradients of $\epsilon_{1r}$ are normalized at line 5 in Algorithm 1, so its scale seems to have no impact on the performance of the proposed method.*

Sorry to not make it unclear. The $\lambda_1$ is after the normalization. As shown in xxx, we can reducing the impact of one compoment by enhance it.

Q1 and Q2 are answered in W1 and W2.

**L1:**  *The computational cost increases due to the iterative multi-objective optimization.* 
extra cost existance, but not too much.

**L2:**   *When we use many functions simultaneously for guidance, some of them might be ignored due to the multi-objective optimization, because any beta can be zero after the optimization.*

Yes, at low probability some of the guidance is ignored but just at specific timesteps