#ifndef MODEL_LR_SCHEDULER_H
#define MODEL_LR_SCHEDULER_H

class LRScheduler{
    public:
        ~LRScheduler(){};

        virtual void step(double& learning_rate, int epoch) = 0;
};

class StepLR : public LRScheduler{
    public:
        StepLR(double gamma, int step_size);

        void step(double& learning_rate, int epoch);

    private:
        double gamma;
        int step_size;
};

#endif // MODEL_LR_SCHEDULER_H