## To gkPe

**W1:**  <1> *In section 2.2, according to the author's definition of image-to-image tasks, it looks like the proposed model is conditioned on a specific image $y_0$. Could SDDM capture the distribution of domain Y instead of a single reference image?*

Yes, SDDM is capable of capturing the distribution of Domain Y using a pretrained Energy function or by utilizing the manifold defined by the entire reference dataset. Our implementation has already achieved excellent results by constructing manifolds of specific image $y_0$, but we can also use energy function to capture the domain information. In fact, our approach has outperformed both StarGAN v2 and EGSDE. I will provide more details on this in W2.


<2>*What is the computation cost for the transfer of given N images in the source domain?*

The computational cost for processing N images of size CxHxW with SDDM is O(NCHW), which is negligible when compared to the computational cost of inferencing neural networks.
To better understand this, let's consider the computational cost of a 3x3 convolution, which is 9CHW.

**W2:** *In 5.1, the GAN models that the author list are pretty old. The GAN model could achieve 13.7 in CelebA-HQ and 16.2 in AFHQ in [1], 7.79 in CelebA-HQ, and 8.04 in FFHQ. It is not fair to compare only with old models.*

We sincerely apologize for not providing detailed information about the FID experiment settings in the paper. Since the diffusion process is slower than GANs, we followed EGSDE's approach and reduced the number of generated images for the FID comparison. We will also include the results of StarGAN v2 for a fair comparision. In the Cat2Dog task, our method outperformed both StarGAN v2 and EGSDE. We will also include the FID results of other datasets in the final version.
[TODO]这里要贴一个实验的表格对比  要最好的FID结果Cat2Dog, 已经完成

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

<1> we will consider adding more tasks for comparision.

<2> We propose a new framework! 我们没有用强力的energy函数, refer EGSDE. and we add some [用上EGSDE, 人脸 打败, 完成].

**Q1:** *Any theoretical justification what manifolds are more suitable in practice?*

Normally we just use manifolds by 经验主义 介绍经验. 如果定量的话, 我们可以考虑在仅考虑一阶矩二阶矩的情况下(当成高斯分布) 使用数据分布和流形对应的分布之间的KL距离来衡量流形对数据分布的影响.

L1: extra computations

## To Kgrj

**W1:** *Several related studies are missed. Due to this, I could not sufficiently judge the novelty as well as significance of this work.*

We propose a general framwork.

<1> Compared with 
Diffusion Visual Counterfactual Explanations 平衡多个guidance 跟EGSDE没有本质区别, Normalization 没有从moo的视角来看问题, We go further to avoid the guidance negatively affect score. 
I don't find this paper, I will cite this paper in the final version.
<2> Compared with
Improving Diffusion Models for Inverse Problems using Manifold Constraints 
This method has strict linear form about the condition (Which is hard for many conditional generation tasks) and use the strong manifold assumption to ensure they has a closed-form to get a 余项到流形上.  while our method follows manifold optimization, use the Restriction function as the bridge for optimization on the manifolds, which can be applied well for more conditional generation tasks.
<3> Compared with Score-SDE
The predictor-corrector method is amazing, but we are totally different. from t to t-1, predictor and corrector are both to keep the distribution of t-1
But in our method we try to reach the local balance point of score, energy guidances.

**W2:**  *Although one of the important advantages of the proposed method is the dynamic balancing of the multiple functions for guidance, this point has not been verified empirically, because the authors used only a single energy function for guidance in the experiments.*

已经有一个, 我们会比较加入多个guidance 函数的影响. 我们的算法在score 和guidance 也会有一个动态平衡的效果.已经在奏效了.

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
