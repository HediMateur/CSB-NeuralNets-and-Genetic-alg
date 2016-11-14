#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <math.h>
#include <list>
#include <fstream>

#include <typeinfo>
#define PI 3.14159265

using namespace std;

struct SNeuron
{
    inline double RandFloat()		   {return (rand())/(RAND_MAX+1.0);}
    inline double RandomClamped()	   {return RandFloat() - RandFloat();}
	int numInputs;
    vector<double>	vecWeight;
    SNeuron(int numInputs);
};
SNeuron::SNeuron(int NumInputs): numInputs(NumInputs+1)

{
	for (int i=0; i<numInputs+1; ++i)
	{
		vecWeight.push_back(RandomClamped());
	}
}

struct SNeuronLayer
{

	int	numNeurons;
	vector<SNeuron> vecNeurons;

	SNeuronLayer(int numNeurons,
				       int numInputsPerNeuron);
};
SNeuronLayer::SNeuronLayer(int numNeurons,
                           int numInputsPerNeuron):	numNeurons(numNeurons)
{
	for (int i=0; i<numNeurons; ++i)

		vecNeurons.push_back(SNeuron(numInputsPerNeuron));
}

class CNeuralNet
{

private:

	int numInputs;

	int	numOutputs;

	int	numHiddenLayers;

	int	neuronsPerHiddenLyr;

	vector<SNeuronLayer> vecLayers;

public:

	CNeuralNet(){
        numInputs	          =	14;
        numOutputs		      =	4;
        numHiddenLayers	    =	3;
        neuronsPerHiddenLyr =	6;
        CreateNet();
	}
	void CreateNet(){

        if (numHiddenLayers > 0)
        {
            vecLayers.push_back(SNeuronLayer(neuronsPerHiddenLyr, numInputs));
            for (int i=0; i<numHiddenLayers-1; ++i)
            {
                vecLayers.push_back(SNeuronLayer(neuronsPerHiddenLyr,
                                         neuronsPerHiddenLyr));
            }


            vecLayers.push_back(SNeuronLayer(numOutputs, neuronsPerHiddenLyr));
        }

        else
        {
            vecLayers.push_back(SNeuronLayer(numOutputs, numInputs));
        }
	}
	vector<double>	GetWeights()const{
        vector<double> weights;

        for (int i=0; i<numHiddenLayers + 1; ++i)
        {

            for (int j=0; j<vecLayers[i].numNeurons; ++j)
            {
                for (int k=0; k<vecLayers[i].vecNeurons[j].numInputs; ++k)
                {
                    weights.push_back(vecLayers[i].vecNeurons[j].vecWeight[k]);
                }
            }
        }

        return weights;
	}
	int GetNumberOfWeights()const{

        int weights = 0;
        for (int i=0; i<numHiddenLayers + 1; ++i)
        {
            for (int j=0; j<vecLayers[i].numNeurons; ++j)
            {
                for (int k=0; k<vecLayers[i].vecNeurons[j].numInputs; ++k)
                weights++;
			}
        }

        return weights;
	}
	void PutWeights(vector<double> &weights){

        int cWeight = 0;
        for (int i=0; i<numHiddenLayers + 1; ++i)
        {
            for (int j=0; j<vecLayers[i].numNeurons; ++j)
            {
                for (int k=0; k<vecLayers[i].vecNeurons[j].numInputs; ++k)
                {
                    vecLayers[i].vecNeurons[j].vecWeight[k] = weights[cWeight++];

                }
            }
        }
        return;
	}
	vector<double> Update(vector<double> &inputs){
        vector<double> outputs;
        int cWeight = 0;
        if (inputs.size() != numInputs)
        {
            return outputs;
        }

        for (int i=0; i<numHiddenLayers + 1; ++i)
        {
            if ( i > 0 )
            {
                inputs = outputs;
            }
            outputs.clear();
            cWeight = 0;
            for (int j=0; j<vecLayers[i].numNeurons; ++j)
            {
                double netinput = 0;
                int	NumInputs = vecLayers[i].vecNeurons[j].numInputs;
                for (int k=0; k<NumInputs - 1; ++k)
                {
                    netinput += vecLayers[i].vecNeurons[j].vecWeight[k] *inputs[cWeight++];
                }
                netinput += vecLayers[i].vecNeurons[j].vecWeight[NumInputs-1] *(-1);
                outputs.push_back(Sigmoid(netinput,1));
                cWeight = 0;
            }
        }
        return outputs;
	}
	inline double Sigmoid(double netinput, double response){
        return ( 1 / ( 1 + exp(-netinput / response)));
	}

};
struct SGenome
{
	vector <double>	vecWeights;
	double dFitness;
	SGenome():dFitness(0){}
    SGenome( vector <double> w, double f): vecWeights(w), dFitness(f){}
    friend bool operator<(const SGenome& lhs, const SGenome& rhs)
	{
		return (lhs.dFitness < rhs.dFitness);
	}
};

class CGenAlg
{
    private:
	vector <SGenome> vecPop;
	int popSize;
	int chromoLength;
	double dTotalFitness;
	double dBestFitness;
	double dAverageFitness;
	double dWorstFitness;
    int dFittestGenome;
	double mutationRate;
	double crossoverRate;
	int	  cGeneration;
	void Crossover(const vector<double> &mum,const vector<double> &dad,vector<double> &baby1,vector<double> &baby2){
        if ( (RandFloat() > crossoverRate) || (mum == dad))
        {
            baby1 = mum;
            baby2 = dad;
            return;
        }
        int cp = RandInt(0, chromoLength - 1);
        for (int i=0; i<cp; ++i)
        {
            baby1.push_back(mum[i]);
            baby2.push_back(dad[i]);
        }
        for (int i=cp; i<mum.size(); ++i)
        {
            baby1.push_back(dad[i]);
            baby2.push_back(mum[i]);
        }
        return;
	}
	void Mutate(vector<double> &chromo){
        for (int i=0; i<chromo.size(); ++i)
        {
            if (RandFloat() < mutationRate)
            {
                chromo[i] += (RandomClamped() * 0.3);
            }
        }
	 }

	SGenome	GetChromoRoulette(){
        double Slice = (double)(RandFloat() * dTotalFitness);
        SGenome TheChosenOne;
        double FitnessSoFar = 0;
        for (int i=0; i<popSize; ++i)
        {
            FitnessSoFar += vecPop[i].dFitness;
            if (FitnessSoFar >= Slice)
            {
                TheChosenOne = vecPop[i];
                break;
            }

        }
        return TheChosenOne;
	}

	vector<SGenome>	GetBestGenome(){
        vector <SGenome> vecP = vecPop;
        vector<SGenome> choosenGenomes;
        int k=0;
        SGenome TheChoosenOne;
        for(int j=0;j<2;j++){
            double HighestSoFar = -9999999999;
            for(int i=0;i<vecP.size();i++){
                if (vecP[i].dFitness > HighestSoFar)
                {
                    HighestSoFar = vecP[i].dFitness;
                    TheChoosenOne = vecP[i];
                    k=i;
                }
             vecP.erase(vecP.begin()+k);
             choosenGenomes.push_back(TheChoosenOne);
            }
        }
        return choosenGenomes;
	}
    void GrabNBest(int NBest,const int NumCopies,vector<SGenome> &Pop){

        while(NBest--)
        {
            for (int i=0; i<NumCopies; ++i)
            {

                Pop.push_back(vecPop[(popSize - 1) - NBest]);


            }
        }
    }
	void CalculateBestWorstAvTot(){
        dTotalFitness = 0;
        double HighestSoFar = 0;
        double LowestSoFar  = 99999999;
        for (int i=0; i<popSize; ++i)
        {
            if (vecPop[i].dFitness > HighestSoFar)
            {
                HighestSoFar = vecPop[i].dFitness;
                dFittestGenome = i;
                dBestFitness = HighestSoFar;
            }
            if (vecPop[i].dFitness < LowestSoFar)
            {
                LowestSoFar = vecPop[i].dFitness;
                dWorstFitness = LowestSoFar;
            }
            dTotalFitness	+= vecPop[i].dFitness;
        }
        dAverageFitness = dTotalFitness / popSize;
	}
	void Reset(){
        dTotalFitness = 0;
        dBestFitness = 0;
        dWorstFitness = 999999999;
        dAverageFitness	= 0;
	}
    public:

    CGenAlg(int popsize,double	MutRat,double	CrossRat,int numweights) :	popSize(popsize),
                                                                            mutationRate(MutRat),
                                                                            crossoverRate(CrossRat),
                                                                            chromoLength(numweights),
                                                                            dTotalFitness(0),
                                                                            cGeneration(0),
                                                                            dFittestGenome(0),
                                                                            dBestFitness(0),
                                                                            dWorstFitness(9999999999),
                                                                            dAverageFitness(0)
    {
        for (int i=0; i<popSize; ++i)
        {
            vecPop.push_back(SGenome());
            for (int j=0; j<chromoLength; ++j)
            {
                vecPop[i].vecWeights.push_back(RandomClamped());
            }
        }
    }

	vector<SGenome>	Epoch(vector<SGenome> &old_pop){
        vecPop = old_pop;
        Reset();
        sort(vecPop.begin(), vecPop.end());
        CalculateBestWorstAvTot();
        vector <SGenome> vecNewPop;
        if (!(1 * 1 % 2))
        {

            GrabNBest(1, 1, vecNewPop);

        }


        while (vecNewPop.size() < popSize)
        {
            SGenome mum = GetBestGenome()[0];
            SGenome dad = GetBestGenome()[1];
            vector<double> baby1, baby2;
            Crossover(mum.vecWeights, dad.vecWeights, baby1, baby2);
            Mutate(baby1);
            Mutate(baby2);
            vecNewPop.push_back(SGenome(baby1, 0));
            vecNewPop.push_back(SGenome(baby2, 0));
        }
        vecPop = vecNewPop;
        return vecPop;
	}

	vector<SGenome>	GetChromos()const{return vecPop;}
	double AverageFitness()const{return dTotalFitness / popSize;}
	double BestFitness()const{return dBestFitness;}

    inline double RandFloat()		   {return (rand())/(RAND_MAX+1.0);}
    inline int	  RandInt(int x,int y) {return rand()%(y-x+1)+x;}
    inline double RandomClamped()	   {return RandFloat() - RandFloat();}
};
class Point{
    protected :
    float x;
    float y;
    public:
    Point(){}
    Point(float x,float y)
    {
        this->x=x;

        this->y=y;
    }
    float Getx(){
        return x;
    }
    float Gety(){
        return y;
    }

};
class Checkpoint;
class Pod;
class Collision{
    public:
    Checkpoint* a;
    Pod* pa;
    Pod* pb;
    //Pod* pb;
    float t;

    Collision(Pod* pa,Checkpoint* a,float t){
        this->pa = pa;
        this->a =a;
        this->t = t;
        this->pb=NULL;
    }
    Collision(Pod* pa,Pod* pb,float t){
        this->pa = pa;
        this->pb =pb;
        this->t = t;
        this->a=NULL;
    }
    /*Collision(Pod* pa,Pod* pb,float t){
        this->pa = pa;
        this->pb =pb;
        this->t = t;
    }*/
};
class Checkpoint:public Point{
    protected:
    float r;
    float vx;
    float vy;
    public:
    Checkpoint(){}
    Checkpoint(float x,float y, float vx,float vy ) : Point(x,y)
    {
        this->vx=vx;
        this->vy=vy;
        this->r=600;

    }
    virtual ~Checkpoint() {}
    float getVx(){
        return vx;
    }
    float getVy(){
        return vy;
    }
    Point closestPointOnline(Point first,Point last){
     float A1 = last.Gety() - first.Gety();
     float B1 = first.Getx() - last.Getx();
     double C1 = A1*first.Getx() + B1*first.Gety();
     double C2 = -B1*this->x + A1*this->y;
     double det = A1*A1 + B1*B1;
     double cx = 0;
     double cy = 0;
     if(det != 0){
        cx = (float)((A1*C1 - B1*C2)/det);
        cy = (float)((A1*C2 +B1*C1)/det);
     }else{
        cx = this->x;
        cy = this->y;
     }
     Point* p = new Point(cx, cy);
     return *p;
}

    float getR()
    {
        return r;
    }

    void setR(float r)
    {
        this->r = r;
    }

    void setVx(float vx)
    {
        this->vx = vx;
    }

    void setVy(float vy)
    {
        this->vy = vy;
    }
};
class Pod:public Checkpoint{
    private :
    bool shield;
    bool bst;
    int thrust;
    int k;
    double targetx;
    double targety;
    float angle;
    int timer;
    int initx;
    CNeuralNet brain;
    int inity;
    int stime;
    double fitness;
    vector <Checkpoint*> checkpoints;
    vector <Checkpoint*> points;
    public:
    int nextCheckPoint;
    int checkpointTaken;
    int GetNumberOfWeights(){
        return brain.GetNumberOfWeights();
    }
    void PutWeights(vector<double> vecWeight){

        brain.PutWeights(vecWeight);
    }
    Pod(){}
    Pod(float x,float y,float vx,float vy,vector<Checkpoint*> checkpoints) : Checkpoint(x, y,vx,vy)
    {
        this->nextCheckPoint=1;
        this->angle=getAngle(checkpoints[nextCheckPoint]);
        this->setR(400);
        this->checkpoints=checkpoints;
        this->checkpointTaken=0;
        this->k = 0;
        this->initx=this->x;
        this->inity=this->y;
        this->timer=0;
        this->stime=10;
        points.push_back(checkpoints[nextCheckPoint]);
    }
    void Reset(){

        this->nextCheckPoint=1;
        this->angle=getAngle(checkpoints[nextCheckPoint]);
        this->x=initx;
        this->y=inity;
        this->checkpoints=checkpoints;
        this->checkpointTaken=0;
        this->k = 0;
        this->timer=0;
        this->stime=10;
        points.clear();
        points.push_back(checkpoints[nextCheckPoint]);
    }
    bool Update()
    {

        vector<double> inputs;
        for(int i=0;i<5;i++)
        {

            if(i>=points.size())
                {
                    inputs.push_back(-1);
                    inputs.push_back(-1);
                }
            else
               {
                    inputs.push_back(points[i]->Getx());
                    inputs.push_back(points[i]->Gety());
               }
        }


        inputs.push_back(x);
        inputs.push_back(y);
        inputs.push_back(shield);
        inputs.push_back(bst);
        vector<double> output = brain.Update(inputs);
        if (output.size() < 4)
        {

            return false;
        }

        targetx = round(output[0]*16000);
        targety = round(output[1]*9000);
        if(output[2]<0.25)
            {
                shield = true;
                thrust=0;
                stime=0;
            }
        else if(output[2]>=0.25&&output[2]<0.5)
            {
                bst = true;
                shield=false;
                thrust = 0;
            }
        else
            {
                thrust = round(output[3]*100);
                shield= false;
            }
	return true;
}

    float getR()
    {
        return r;
    }
    Collision* collisionWithCheckpoint(Checkpoint* p){
        float distance = (this->x-p->Getx())*(this->x-p->Getx())+(this->y-p->Gety())*(this->y-p->Gety());
        float rayon = (p->getR()+this->r)*(p->getR()+this->r);
        if(distance<=rayon)
        {

            return  new Collision(this,p,0.0);
        }
        float x = this->x - p->Getx();
        float y = this->y - p->Gety();
        Point* pOneRef = new Point(x, y);
        float vx = this->vx - p->getVx();
        float vy = this->vy - p->getVy();
        Point* a = new Point(x,y);
        Point* b=new Point(x+vx,y+vy);
        Point d = p->closestPointOnline(*a,*b);
        double clousestDistance= (d.Getx()-p->Getx())*(d.Getx()-p->Getx())+(d.Gety()-p->Gety())*(d.Gety()-p->Gety());
        if(clousestDistance<=rayon){
            double backdist = sqrt(r - clousestDistance);
            double movementvectorlength = sqrt(this->vx*this->vx + this->vy*this->vy);
            double c_x = d.Getx() - backdist * (this->vx / movementvectorlength);
            double c_y = d.Gety() - backdist * (this->vy / movementvectorlength);
            float dist = (pOneRef->Getx()-c_x)*(pOneRef->Getx()-c_x)+(pOneRef->Gety()-c_y)*(pOneRef->Gety()-c_y);
            if(dist>movementvectorlength)
                return NULL;
            float time = dist/movementvectorlength;

            return new Collision(this,p,time);
        }
        else
            return NULL;

    }
    Collision* collisionWithPod(Pod* p){
        float distance = (this->x-p->Getx())*(this->x-p->Getx())+(this->y-p->Gety())*(this->y-p->Gety());
        float rayon = (p->getR()+this->r)*(p->getR()+this->r);
        if(distance<=rayon)
        {

            return  new Collision(this,p,0.0);
        }
        float x = this->x - p->Getx();
        float y = this->y - p->Gety();
        Point* pOneRef = new Point(x, y);
        float vx = this->vx - p->getVx();
        float vy = this->vy - p->getVy();
        Point* a = new Point(x,y);
        Point* b=new Point(x+vx,y+vy);
        Point d = p->closestPointOnline(*a,*b);
        double clousestDistance= (d.Getx()-p->Getx())*(d.Getx()-p->Getx())+(d.Gety()-p->Gety())*(d.Gety()-p->Gety());
        if(clousestDistance<=rayon){
            double backdist = sqrt(r - clousestDistance);
            double movementvectorlength = sqrt(this->vx*this->vx + this->vy*this->vy);
            double c_x = d.Getx() - backdist * (this->vx / movementvectorlength);
            double c_y = d.Gety() - backdist * (this->vy / movementvectorlength);
            float dist = (pOneRef->Getx()-c_x)*(pOneRef->Getx()-c_x)+(pOneRef->Gety()-c_y)*(pOneRef->Gety()-c_y);
            if(dist>movementvectorlength)
                return NULL;
            float time = dist/movementvectorlength;
            return new Collision(this,p,time);
        }
        else
            return NULL;

    }
    float getAngle(Checkpoint* p)
    {
      double dy = p->Gety()-this->y ;
      double dx = p->Getx()-this->x ;
      double theta = atan2(dy,dx)*180/PI;
      //if(this->nextCheckPoint==2)
      //cout<<this->angle<<"theta";
      //if(theta<0)
      //theta=360.0+theta;

      return theta;
    }
    float getDiffAng(Checkpoint* p)
    {

        float ang = this->getAngle(p);

        float r = this->angle <= ang ? ang - this->angle : 360.0 - this->angle + ang;
        float l = this->angle >= ang ? this->angle - ang : this->angle + 360.0 - ang;


        if (r < l) {
        return r;
        } else {

            return -l;
        }
    }
    void bounce(Pod* p){
        float m1 = this->shield ? 10 : 1;
        float m2 = p->shield ? 10 : 1;
        float mcoeff = (m1 + m2) / (m1 * m2);
        float nx = this->x - p->x;
        float ny = this->y - p->y;


        float nxnysquare = nx*nx + ny*ny;

        float dvx = this->vx - p->vx;
        float dvy = this->vy - p->vy;

        float product = nx*dvx + ny*dvy;
        float fx = (nx * product) / (nxnysquare * mcoeff);
        float fy = (ny * product) / (nxnysquare * mcoeff);

        this->vx -= fx / m1;
        this->vy -= fy / m1;
        float vx2 = p->getVx()+fx / m2;
        float vy2 = p->getVy()+fy / m2;
        float impulse = sqrt(fx*fx + fy*fy);
        if (impulse < 120.0) {
            fx = fx * 120.0 / impulse;
            fy = fy * 120.0 / impulse;
        }
        this->vx -= fx / m1;
        this->vy -= fy / m1;
        vx2 += fx / m2;
        vy2 += fy / m2;
        p->setVx(vx2);
        p->setVy(vy2);

        this->moveTo(1.0);
        p->moveTo(1.0);

    }

    void bounce(Checkpoint* c) {
        this->bouceWithCheck();

    }
    int getTime(){
        return timer;
    }
    void turn()
    {
        //cout<<p->Getx()<<endl;
        Checkpoint* c = new Checkpoint(targetx,targety,0,0);

        float a = this->getDiffAng(c);
        //if(this->nextCheckPoint==2)
        //cout<<a<<endl;
        if(a>18){
            a=18;
        }
        else if(a<-18){
            a=-18;
        }
       //     cout<<this->angle<<endl;
            this->angle+=a;
        if(this->angle>=360.0)
            this->angle-=360.0;
        else if (this->angle<0.0){
            this->angle+=360.0;
        }
    }
    void speed()
    {

        if(stime<3)
        {
            cout<<"shield"<<endl;
            stime++;
            return;
        }
        else if (bst&&k<1)
        {
            cout<<"boost"<<endl;
            float angleRadiant = this->angle *PI/180;
            this->vx += cos(angleRadiant)*650;
            this->vy += sin(angleRadiant)*650;
            k+=1;
        }
        else
        {
            float angleRadiant = this->angle *PI/180;
            this->vx += cos(angleRadiant)*thrust;
            this->vy += sin(angleRadiant)*thrust;
        }
    }
    void moveTo(float t)
    {
        //cout<<vx<<endl;
        this->x +=this->vx*t;
        this->y +=this->vy*t;
        timer++;
    }
    void bouceWithCheck()
    {
        this->nextCheckPoint +=1;
        if(nextCheckPoint==checkpoints.size())
            this->nextCheckPoint == 0;
        this->checkpointTaken+=1;
        std::ofstream outfile ("test.txt",ios::app);
        outfile <<"checkpointTaken "<< std::endl;
        outfile.close();

        this->timer=0;
        bool tester = false;
        for (int i =0; i<points.size();i++)
            {
                if(points[i]->Getx()==checkpoints[nextCheckPoint]->Getx()&&points[i]->Gety()==checkpoints[nextCheckPoint]->Gety())
                tester = true;
            }
        if(!tester)
            points.push_back(checkpoints[nextCheckPoint]);
        //std::ofstream outfile ("test.txt",ios::app);

    }
    float score()
    {
        if(timer>=100)
            return 0;
        else{
            return checkpointTaken*10-timer;
        }
    }
    void afterMoving()
    {
        this->x = round(this->x);
        this->y=round(this->y);
        this->vx=trunc(this->vx*0.85);
        this->vy=trunc(this->vy*0.85);
        cout<<x<<" "<<y<< " "<<thrust<<endl;
        std::ofstream outfile ("test.txt",ios::app);
        outfile << x<<" "<<y<< std::endl;
        outfile.close();

    }
    /*void simulate(Checkpoint* p,int thrust)
    {

        turn(p);
        speed(thrust);
        moveTo(0);
        afterMoving();
    }*/

};

class PodController
{

    private:

	vector<SGenome> vecThePopulation;
    vector<Pod*> pods;
	vector<Checkpoint*> checkpoints;
    CGenAlg* pGA;
    int numWeightsInNN;
	vector<double>		   vecAvFitness;
	vector<double>		   vecBestFitness;
	int iTicks;
	int	iGenerations;

    public:

	PodController(vector<Pod*> pods, vector<Checkpoint*> checkpoints){
        this->pods = pods;
        this->checkpoints = checkpoints;
        this->iTicks = 0;
        this->pGA = NULL;
        this->iGenerations = 0;
        this->numWeightsInNN = pods[0]->GetNumberOfWeights();
        this->pGA = new CGenAlg(pods.size(),0.1,0.7,numWeightsInNN);
        this->vecThePopulation = pGA->GetChromos();
        for(int i=0;i<pods.size();i++)
            pods[i]->PutWeights(vecThePopulation[i].vecWeights);
	}

    bool Update(){
        for(int j=0;j<pods.size();j+=2){
            iTicks=0;
            std::ofstream outfile ("test.txt",ios::app);
            outfile <<"Pod"<<j<<"vs Pod "<<j+1<< std::endl;
            outfile.close();
            while(iTicks++<2000){
                for(int i=j;i<j+2;i++){
                    if(!pods[i]->Update()){
                        return false;
                    }
                    pods[i]->turn();
                    pods[i]->speed();
                    if(pods[i]->getTime()==100)
                        iTicks = 2001;

                }
                play(pods,checkpoints,j);
                for(int i=j;i<j+2;i++){

                    vecThePopulation[i].dFitness = pods[i]->score();
                }
            }
        }

        vecAvFitness.push_back(pGA->AverageFitness());
        vecBestFitness.push_back(pGA->BestFitness());
        ++iGenerations;
        iTicks = 0;
        vecThePopulation = pGA->Epoch(vecThePopulation);
        std::ofstream outfile ("test.txt",ios::app);
        outfile <<"Another generation "<< std::endl;
        outfile.close();
        float score=0;
        for(int i = 0;i<pods.size();i++){
                pods[i]->PutWeights(vecThePopulation[i].vecWeights);
                pods[i]->Reset();
            }

        return true;
    }
    ~PodController() {
        if(pGA)
        delete pGA;
    }

    void play(vector<Pod*> pods, vector<Checkpoint*> checkpoints,int k) {
    float t = 0.0;
    while (t < 1.0) {
        Collision* firstCollision = NULL;
        for (int i = k; i < k+2; ++i) {
            for (int j = i + 1; j < k+2; ++j) {
                Collision* col = pods[i]->collisionWithPod(pods[j]);
                    if (col != NULL && col->t + t < 1.0 && (firstCollision == NULL || col->t < firstCollision->t)) {

                    firstCollision = col;
                }
            }
            int k = pods[i]->nextCheckPoint;
            if(k==checkpoints.size())
            {
                    k=0;
                    pods[i]->nextCheckPoint =k;
            }
            //cout<<k<<" "<<checkpoints[k]->Getx()<<endl;

            Collision* col = pods[i]->collisionWithCheckpoint(checkpoints[k]);

            if (col != NULL && col->t + t < 1.0 && (firstCollision == NULL || col->t < firstCollision->t)) {
                firstCollision = col;
            }

        }
        if (firstCollision == NULL) {
            for (int i = k; i < k+2; ++i) {
                pods[i]->moveTo(1.0 - t);
            }
            t = 1.0;
        } else {
            for (int i = k; i < k+2; ++i) {
                pods[i]->moveTo(firstCollision->t-t);
            }
            if(firstCollision->a==NULL){
            firstCollision->pa->bounce(firstCollision->pb);

            }
            else{
            firstCollision->pa->bounce(firstCollision->a);
            }
            t += firstCollision->t;

        }
    }

    for (int i = k; i < k+2; ++i) {

            pods[i]->afterMoving();
        }
    }


};



/**
 * Auto-generated code below aims at helping you parse
 * the standard input according to the problem statement.
 **/
using namespace std;

int main()
{

    // game loop
        float x=3474;
        float y=6733;
        float nextCheckpointX=9417; // x position of the next check point
        float nextCheckpointY=7257; // y position of the next check point
        //int nextCheckpointDist; // distance to the next checkpoint
        int nextCheckpointAngle=0+180; // angle between your pod orientation and the direction of the next checkpoint
        //cin >> x >> y >> nextCheckpointX >> nextCheckpointY >> nextCheckpointDist >> nextCheckpointAngle; cin.ignore();
        //int opponentX;
        //int opponentY;
        //cin >> opponentX >> opponentY; cin.ignore();
        Checkpoint* p0 = new Checkpoint(3472,7233,0,0);
        Checkpoint* p1 = new Checkpoint(nextCheckpointX,nextCheckpointY,0,0);
        Checkpoint* p2 = new Checkpoint(5977,4262,0,0);
        Checkpoint* p3 = new Checkpoint(14680,1432,0,0);
        //Checkpoint* p4 = new Checkpoint(10698,2303,0,0);

        vector <Checkpoint*> checkpoints;
        checkpoints.push_back(p0);
        checkpoints.push_back(p1);
        checkpoints.push_back(p2);
        checkpoints.push_back(p3);
        //checkpoints.push_back(p4);
        Pod* po1 = new Pod(3474,6733,0,0,checkpoints);
        Pod* po2 = new Pod(3470,7733,0,0,checkpoints);
        Pod* po3 = new Pod(2474,6733,0,0,checkpoints);
        Pod* po4 = new Pod(2470,7733,0,0,checkpoints);
        Pod* po5 = new Pod(1474,6733,0,0,checkpoints);
        Pod* po6 = new Pod(1470,7733,0,0,checkpoints);
        Pod* po7 = new Pod(474,6733,0,0,checkpoints);
        Pod* po8 = new Pod(470,7733,0,0,checkpoints);
        Pod* po9 = new Pod(3474,5733,0,0,checkpoints);
        Pod* po10 = new Pod(3470,6733,0,0,checkpoints);
        Pod* po11 = new Pod(3474,4733,0,0,checkpoints);
        Pod* po12 = new Pod(3470,5733,0,0,checkpoints);
        Pod* po13 = new Pod(3474,3733,0,0,checkpoints);
        Pod* po14 = new Pod(3470,4733,0,0,checkpoints);
        vector <Pod*> pods;
        pods.push_back(po1);
        pods.push_back(po2);
        pods.push_back(po3);
        pods.push_back(po4);
        pods.push_back(po5);
        pods.push_back(po6);
        pods.push_back(po7);
        pods.push_back(po8);
        pods.push_back(po9);
        pods.push_back(po10);
        pods.push_back(po11);
        pods.push_back(po12);
        pods.push_back(po13);
        pods.push_back(po14);
        PodController* podController = new PodController(pods,checkpoints);
        for(int i=0;i<10;i++)
        podController->Update();

        //for(int i=0; i<20; i++)
        //po->simulate(*p,thrust);
       // Pod* pod = static_cast<Pod*>(p);
        // Write an action using cout. DON'T FORGET THE "<< endl"
        // To debug: cerr << "Debug messages..." ;


        // You have to output the target position
        // followed by the power (0 <= thrust <= 100) or "BOOST"
        // i.e.: "x y thrust"
        //cout << nextCheckpointX << " " << nextCheckpointY << " 80" << endl;


}
