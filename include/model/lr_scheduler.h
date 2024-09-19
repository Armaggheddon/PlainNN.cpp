#ifndef MODEL_LR_SCHEDULER_H
#define MODEL_LR_SCHEDULER_H

/**
 * @brief Abstract class for learning rate schedulers. New learning
 * rate schedulers should inherit from this class and implement all
 * of its methods.
 */
class LRScheduler{
    public:
        ~LRScheduler(){};

        /**
         * @brief Step the learning rate
         * 
         * @param learning_rate The learning rate to step
         * @param epoch The current epoch
         * 
         * @note This function should modify the learning rate in place
         * and is called at the end of each epoch.
         */
        virtual void step(double& learning_rate, int epoch) = 0;
};

/**
 * @brief Step learning rate scheduler
 * 
 * This scheduler steps the learning rate by gamma every step_size epochs.
 */
class StepLR : public LRScheduler{
    public:
        StepLR(double gamma, int step_size);

        void step(double& learning_rate, int epoch);

    private:
        double gamma;
        int step_size;
};

#endif // MODEL_LR_SCHEDULER_H