#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/contrib/scenetext.hpp"

#include <iostream>
#include <stdlib.h>
#include <limits.h>
#include <sstream>

using namespace cv;
using namespace std;

namespace cv
{
    /////////////////////////////
    //
    // Region1D methods
    //
    /////////////////////////////

    Region::Region()
    {

    }

    Region::Region(Point _p, int h)
    {
        start = _p;
        bounds.x = _p.x;
        bounds.y = _p.y;
        bounds.width = 1;
        bounds.height = 1;
        area = 1;
        perimeter = 4;
        euler = 0;
        imageh = h;
        crossings = new int[imageh + 1];
        memset(crossings, 0, imageh * 4);
        crossings[_p.y] = 2;
    }

    Region::Region(Point _s, Rect _b, int _a, int _p, int _e, int* _c, int h)
    {
        start = _s;
        bounds = _b;
        area = _a;
        perimeter = _p;
        euler = _e;
        imageh = h;
        crossings = new int[imageh + 1];
        memset(crossings, 0, imageh * 4);
        memcpy(crossings, _c, imageh * 4);
    }

    Region::~Region()
    {
        //delete[] crossings;
    }

    void Region::Attach(Region* _extra, int _borderLength, int _p0y, int _hn)
    {
        if (start != _extra->start)
        {
            bounds |= _extra->bounds;
            area += _extra->area;
            perimeter += _extra->perimeter - 2 * _borderLength;
            euler += _extra->euler;
            for(int i = bounds.y; i < bounds.y + bounds.height; i++)
            {
                crossings[i] += _extra->crossings[i];
            }
            crossings[_p0y] -= 2 * _hn;
        }
    }

    void Region::CorrectEuler(int _delta)
    {
        euler += _delta;
    }

    Rect Region::Bounds()
    {
        return bounds;
    }

    Point Region::Start()
    {
        return start;
    }

    int Region::Area()
    {
        return area;
    }

    int Region::Perimeter()
    {
        return perimeter;
    }

    int Region::Euler()
    {
        return euler;
    }

    int Region::Crossings(int _y)
    {
        return crossings[_y];
    }

    int* Region::AllCrossings()
    {
        return crossings;
    }

    int Region::BoundsArea()
    {
        return bounds.width * bounds.height;
    }

    /////////////////////////////
    //
    // Region1D methods
    //
    /////////////////////////////

    Region1D::Region1D(unsigned _p, unsigned _t, int h, int w)
    {
        start = _p;
        thresh = _t;
        bounds.x = _p % w;
        bounds.y = _p / w;
        bounds.width = 1;
        bounds.height = 1;
        area = 1;
        perimeter = 4;
        euler = 0;
        imageh = h;
        crossings = NULL;

        memset(&crossings_small[0], 0, SMALL_SIZE * 4);

        top_of_small = bounds.y - SMALL_SIZE_MIDDLE;
        if (bounds.y < SMALL_SIZE_MIDDLE)
        {
            top_of_small = 0;
        }
        if (top_of_small + SMALL_SIZE > imageh)
        {
            top_of_small = imageh - SMALL_SIZE;
        }
        crossings_small[bounds.y - top_of_small] = 2;
    }

    Region1D::Region1D(unsigned _s, unsigned _t, Rect _b, int _a, int _p, int _e, int* _c, int h)
    {
        start = _s;
        thresh = _t;
        bounds = _b;
        area = _a;
        perimeter = _p;
        euler = _e;
        imageh = h;
        crossings = new int[imageh + 1];
        memset(crossings, 0, imageh * 4);
        memcpy(crossings, _c, imageh * 4);
    }

    Region1D::~Region1D()
    {
        if (top_of_small == INT_MAX)
        {
            delete[] crossings;
        }
    }

    unsigned Region1D::Start()
    {
        return start;
    }

    unsigned Region1D::Threshold()
    {
        return thresh;
    }

    void Region1D::CorrectEuler(int _delta)
    {
        euler += _delta;
    }

    Rect Region1D::Bounds()
    {
        return bounds;
    }

    int Region1D::Area()
    {
        return area;
    }

    int Region1D::Perimeter()
    {
        return perimeter;
    }

    int Region1D::Euler()
    {
        return euler;
    }

    void Region1D::Attach(Region1D* _extra, int _borderLength, int _p0y, int _hn)
    {
        if (start != _extra->start)
        {
            thresh = thresh > _extra->thresh ? thresh : _extra->thresh;
            bounds |= _extra->bounds;
            area += _extra->area;
            perimeter += _extra->perimeter - 2 * _borderLength;
            euler += _extra->euler;

            if ((bounds.height > SMALL_SIZE) || (bounds.y <= top_of_small) || (bounds.y + bounds.height >= top_of_small + SMALL_SIZE))
            {
                if (top_of_small != INT_MAX)
                {
                    crossings = new int[imageh];
                    memset(crossings, 0, imageh * 4);
                    memcpy(&crossings[top_of_small], &crossings_small[0], SMALL_SIZE * 4);

                    top_of_small = INT_MAX;
                }
            }

            if (top_of_small != INT_MAX)
            {
                for(int i = bounds.y; i < bounds.y + bounds.height; i++)
                {
                    crossings_small[i - top_of_small] += _extra->Crossings(i);
                }
                crossings_small[_p0y - top_of_small] -= 2 * _hn;
            }
            else
            {
                for(int i = bounds.y; i < bounds.y + bounds.height; i++)
                {
                    crossings[i] += _extra->Crossings(i);
                }
                crossings[_p0y] -= 2 * _hn;
            }
        }
    }

    void Region1D::AttachPoint(unsigned _p, unsigned _t, int w, int _borderLength, int _hn)
    {
        int px = _p % w;
        int py = _p / w;
        bounds |= Rect(px, py, 1, 1);
        thresh = _t;
        area++;
        perimeter += 4 - 2 * _borderLength;
        //euler += 0;

        if ((bounds.height > SMALL_SIZE) || (bounds.y <= top_of_small) || (bounds.y + bounds.height >= top_of_small + SMALL_SIZE))
        {
            if (top_of_small != INT_MAX)
            {
                crossings = new int[imageh];
                memset(crossings, 0, imageh * 4);
                memcpy(&crossings[top_of_small], &crossings_small[0], SMALL_SIZE * 4);

                top_of_small = INT_MAX;
            }
        }

        if (top_of_small != INT_MAX)
        {
            crossings_small[py - top_of_small] += 2;
            crossings_small[py - top_of_small] -= 2 * _hn;
        }
        else
        {
            crossings[py] += 2;
            crossings[py] -= 2 * _hn;
        }
    }

    int Region1D::TopOfSmall()
    {
        return top_of_small;
    }

    int* Region1D::AllCrossings()
    {
        if (top_of_small != INT_MAX)
        {
            crossings = new int[imageh + 1];
            memset(crossings, 0, imageh * 4);
            memcpy(&crossings[top_of_small], &crossings_small[0], SMALL_SIZE * 4);
            top_of_small = INT_MAX;
        }
        return crossings;
    }

    /////////////////////////////
    //
    // SceneTextLocalizer methods
    //
    /////////////////////////////

    SceneTextLocalizer::SceneTextLocalizer()
    {

    }

    SceneTextLocalizer::SceneTextLocalizer(Mat _image, int _thresh)
    {
        _originalImage = _image;
        threshValue = _thresh;
    }

    Point* SceneTextLocalizer::uf_Find(Point* _x, Point** _parents)
    {
        static Point stub = Point (-1, -1);
        if (_parents[_x->x][_x->y].x == -1)
        {
            return &stub;
        }
        while(_parents[_x->x][_x->y] != *_x)
        {
            _x = &_parents[_x->x][_x->y];
        }
        return _x;
    }

    inline unsigned* SceneTextLocalizer::uf_Find1D(unsigned* _x, unsigned* _parents)
    {
        // UINT_MAX means stub
        // UINT_MAX-1 means that point is root of some region
        while(_parents[*_x] < UINT_MAX - 1)
        {
            _x = &_parents[*_x];
        }
        return _parents[*_x] == UINT_MAX ? &_parents[*_x] : _x;
    }

    inline bool SceneTextLocalizer::uf_CheckStub(unsigned* _x, unsigned* _parents)
    {
        while(_parents[*_x] < UINT_MAX - 1)
        {
            _x = &_parents[*_x];
        }
        return _parents[*_x] != UINT_MAX;
    }

    set<Region, RegionComp> SceneTextLocalizer::GroundTruth()
    {
        double t = (double)getTickCount();

        uchar thresholdValue = 100;
        uchar maxValue = 255;
        uchar middleValue = 192;
        uchar zeroValue = 0;
        Scalar middleScalar(middleValue);
        Scalar zeroScalar(zeroValue);

        static int neigborsCount = 4;
        static int dx[] = {-1,  0, 0, 1};
        static int dy[] = { 0, -1, 1, 0};
        int di, rx, ry;
        int perimeter;

        Mat _bwImage(_originalImage.size(), CV_8UC1);
        cvtColor(_originalImage, _bwImage, CV_RGB2GRAY);
        threshold(_bwImage, _bwImage, thresholdValue, maxValue, THRESH_BINARY_INV);
        Mat bwImage(_bwImage.rows + 2, _bwImage.cols + 2, _bwImage.type());
        copyMakeBorder(_bwImage, bwImage, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0, 0, 0));

        int regionsCount = 0;
        int totalPixelCount = bwImage.rows * bwImage.cols;
        Point seedPoint;
        Rect rectFilled;
        int valuesSum, q1, q2, q3;
        bool p00, p10, p01, p11;

        set<Region, RegionComp> retval;

        for(int i = 0; i < totalPixelCount; i++)
        {
            if (bwImage.data[i] == maxValue)
            {
                seedPoint.x = i % bwImage.cols;
                seedPoint.y = i / bwImage.cols;

                if ((seedPoint.x == 0) || (seedPoint.y == 0) || (seedPoint.x == bwImage.cols - 1) || (seedPoint.y == bwImage.rows - 1))
                {
                    continue;
                }

                regionsCount++;

                size_t pixelsFilled = floodFill(bwImage, seedPoint, middleScalar, &rectFilled);

                perimeter = 0;
                q1 = 0; q2 = 0; q3 = 0;

                int crossings[bwImage.rows];
                memset(&crossings[0], 0, 4 * bwImage.rows);

                for(ry = rectFilled.y; ry <= rectFilled.y + rectFilled.height; ry++)
                {
                    for(rx = rectFilled.x; rx <= rectFilled.x + rectFilled.width; rx++)
                    {
                        if ((bwImage.at<uint8_t>(ry, rx - 1) != bwImage.at<uint8_t>(ry, rx)) && (bwImage.at<uint8_t>(ry, rx - 1) + bwImage.at<uint8_t>(ry, rx) == middleValue + zeroValue))
                        {
                            crossings[ry]++;
                        }

                        if (bwImage.at<uint8_t>(ry, rx) == middleValue)
                        {
                            for(di = 0; di < neigborsCount; di++)
                            {
                                int xNew = rx + dx[di];
                                int yNew = ry + dy[di];

                                if (bwImage.at<uint8_t>(yNew, xNew) == zeroValue)
                                {
                                    perimeter++;
                                }
                            }
                        }

                        p00 = bwImage.at<uint8_t>(ry, rx) == middleValue;
                        p01 = bwImage.at<uint8_t>(ry, rx - 1) == middleValue;
                        p10 = bwImage.at<uint8_t>(ry - 1, rx) == middleValue;
                        p11 = bwImage.at<uint8_t>(ry - 1, rx - 1) == middleValue;
                        valuesSum = p00 + p01 + p10 + p11;

                        if (valuesSum == 1) q1++; else
                        if (valuesSum == 3) q2++; else
                        if ((valuesSum == 2) && (p00 == p11)) q3++;
                    }
                }

                q1 = q1 - q2 + 2 * q3;
                if (q1 % 4 != 0)
                {
                    exit(0);
                }
                q1 /= 4;

                Region _r(seedPoint, Rect(rectFilled.x - 1, rectFilled.y - 1, rectFilled.width, rectFilled.height), pixelsFilled, perimeter, q1, &crossings[0], bwImage.rows);
                retval.insert(_r);

                floodFill(bwImage, seedPoint, zeroScalar);

                //rectangle(originalImage, rectFilled, zeroScalar);
            }
        }

        t = (double)getTickCount() - t;
        cout << "Working time: " << t * 1000. / getTickFrequency() << " ms" << endl;

        return retval;
    }

    set<Region1D, Region1DComp> SceneTextLocalizer::MatasLike()
    {
        double t = (double)getTickCount();

        Mat originalImage(_originalImage.rows + 2, _originalImage.cols + 2, _originalImage.type());
        copyMakeBorder(_originalImage, originalImage, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(255, 255, 255));

        Mat bwImage(originalImage.size(), CV_8UC1);

        static int neighborsCount = 4;
        static int dx[] = {-1,  0, 0, 1};
        static int dy[] = { 0, -1, 1, 0};
        int di;

        int i, j, k;

        cvtColor(originalImage, bwImage, CV_RGB2GRAY);

        static int thresh_start = 0;
        static int thresh_end = 101;
        static int thresh_step = 1; // see "FEATURE" comments

        // FEATURE: change ibins to ibins / thresh_step and add some memset
        const int ibins = 256;
        unsigned hist[ibins] = { 0 };
        for(i = 0; i < bwImage.rows; i++)
        {
            const uchar* bwImageRow = bwImage.ptr<uchar>(i);
            for(j = 0; j < bwImage.cols; j++)
            {
                // FEATURE: change bwImageRow[j] to bwImageRow[j] / thresh_step in the next line
                hist[bwImageRow[j]]++;
            }
        }

        long nextIndexes[ibins];
        unsigned* sortedPoints = new unsigned[bwImage.rows * bwImage.cols];
        nextIndexes[0] = 0;
        for(int hi=1; hi<ibins; hi++)
        {
            nextIndexes[hi] = nextIndexes[hi-1] + hist[hi-1];
        }

        unsigned* ranksArray = new unsigned[bwImage.rows * bwImage.cols];

        static unsigned stub = UINT_MAX;
        unsigned* parentsArray = new unsigned[bwImage.rows * bwImage.cols];
        std::fill(parentsArray, parentsArray + bwImage.rows * bwImage.cols, stub);

        Region1D** regionsArray;
        regionsArray = new Region1D*[bwImage.rows * bwImage.cols];
        for(i = 0; i < bwImage.rows * bwImage.cols; i++)
        {
            regionsArray[i] = NULL;
        }
        //memset(&regionsArray[0], 0, bwImage.rows * bwImage.cols * sizeof(Region1D*));

        for(i = 0; i < bwImage.rows; i++)
        {
            const uchar* bwImageRow = bwImage.ptr<uchar>(i);
            for(j = 0; j < bwImage.cols; j++)
            {
                // FEATURE: change bwImageRow[j] to bwImageRow[j] / thresh_step in the next two lines
                sortedPoints[nextIndexes[bwImageRow[j]]] = i * bwImage.cols + j;
                nextIndexes[bwImageRow[j]]++;
            }
        }

        int thresh;

        bool changed = false;
        bool is_any_neighbor[3][3];
        is_any_neighbor[1][1] = false;
        int neighborsInRegions = 0, horizontalNeighbors = 0;
        int q1 = 0, q2 = 0, q3 = 0;
        int q10 = 0, q20 = 0, q30 = 0;
        int qtemp = 0;
        int point_rank, neighbor_rank;
        int x_new, y_new;

        unsigned ptemp;
        unsigned first_point_id;
        unsigned p_new, p0, p1;
        unsigned proot_p, p1root_p;

        for(thresh = thresh_start; thresh < thresh_end; thresh += thresh_step)
        {
            first_point_id = thresh == 0 ? 0 : nextIndexes[thresh - 1];
            for(k = first_point_id; k < nextIndexes[thresh]; k++)
            {
                p0 = sortedPoints[k];

                // Surely point when accessed for the first time is not in any region
                // Setting parent, rank, creating region (uf_makeset)
                parentsArray[p0] = UINT_MAX - 1;
                ranksArray[p0] = 0;

                // We don't create a new object for any point at start anymore
                // regionsArray[p0] = new Region1D(p0, bwImage.rows, bwImage.cols);
                proot_p = p0;

                changed = false;
                is_any_neighbor[1][1] = false;
                q1 = 0; q2 = 0; q3 = 0;
                q10 = 0; q20 = 0; q30 = 0;
                qtemp = 0;

                // other loop
                // ddx=-1 ddy=-1
                ptemp = p0 - 1 - bwImage.cols;
                is_any_neighbor[0][0] = uf_CheckStub(&ptemp, parentsArray);

                // ddx=-1 ddy=0
                ptemp = p0 - 1;
                is_any_neighbor[0][1] = uf_CheckStub(&ptemp, parentsArray);

                // ddx=-1 ddy=1
                ptemp = p0 - 1 + bwImage.cols;
                is_any_neighbor[0][2] = uf_CheckStub(&ptemp, parentsArray);

                // ddx=0 ddy=-1
                ptemp = p0 - bwImage.cols;
                is_any_neighbor[1][0] = uf_CheckStub(&ptemp, parentsArray);

                // ddx=0 ddy=0
                qtemp = is_any_neighbor[0][0] + is_any_neighbor[0][1] + is_any_neighbor[1][0]; // + is_any_neighbor[1][1] == false
                if (qtemp == 0)
                {
                    q1++;
                }
                else if (qtemp == 1)
                {
                    q10++;
                    q3 += is_any_neighbor[0][0];
                }
                else if (qtemp == 2)
                {
                    q30 += !is_any_neighbor[0][0];
                    q2++;
                }
                else if (qtemp == 3)
                {
                    q20++;
                }

                // ddx=0 ddy=1
                ptemp = p0 + bwImage.cols;
                is_any_neighbor[1][2] = uf_CheckStub(&ptemp, parentsArray);
                qtemp = is_any_neighbor[0][1] + is_any_neighbor[0][2] + is_any_neighbor[1][2]; // + is_any_neighbor[1][1] == false
                if (qtemp == 0)
                {
                    q1++;
                }
                else if (qtemp == 1)
                {
                    q10++;
                    q3 += is_any_neighbor[0][2];
                }
                else if (qtemp == 2)
                {
                    q30 += is_any_neighbor[1][2] == is_any_neighbor[0][1];
                    q2++;
                }
                else if (qtemp == 3)
                {
                    q20++;
                }

                // ddx=1 ddy=-1
                ptemp = p0 + 1 - bwImage.cols;
                is_any_neighbor[2][0] = uf_CheckStub(&ptemp, parentsArray);

                // ddx=1 ddy=0
                ptemp = p0 + 1;
                is_any_neighbor[2][1] = uf_CheckStub(&ptemp, parentsArray);
                qtemp = is_any_neighbor[1][0] + is_any_neighbor[2][0] + is_any_neighbor[2][1]; // + is_any_neighbor[1][1] == false
                if (qtemp == 0)
                {
                    q1++;
                }
                else if (qtemp == 1)
                {
                    q10++;
                    q3 += is_any_neighbor[2][0];
                }
                else if (qtemp == 2)
                {
                    q30 += is_any_neighbor[1][0] == is_any_neighbor[2][1];
                    q2++;
                }
                else if (qtemp == 3)
                {
                    q20++;
                }

                // ddx=1 ddy=1
                ptemp = p0 + 1 + bwImage.cols;
                is_any_neighbor[2][2] = uf_CheckStub(&ptemp, parentsArray);
                qtemp = is_any_neighbor[1][2] + is_any_neighbor[2][1] + is_any_neighbor[2][2]; // + is_any_neighbor[1][1] == false
                if (qtemp == 0)
                {
                    q1++;
                }
                else if (qtemp == 1)
                {
                    q10++;
                    q3 += is_any_neighbor[2][2];
                }
                else if (qtemp == 2)
                {
                    q30 += !is_any_neighbor[2][2];
                    q2++;
                }
                else if (qtemp == 3)
                {
                    q20++;
                }
                // end loop

                qtemp = (q1 - q2 + q3 * 2) - (q10 - q20 + q30 * 2);

                if (qtemp % 4 != 0)
                {
                    //printf("Non-integer Euler number");
                    exit(0);
                }
                qtemp /= 4;

                bool neighbors_tested = false;

                for(di = 0; di < neighborsCount; di++)
                {
                    x_new = p0 % bwImage.cols + dx[di];
                    y_new = p0 / bwImage.cols + dy[di];
                    p_new = p0 + dx[di] + dy[di] * bwImage.cols;

                    // TODO: implement corresponding function?
                    if ((x_new < 0) || (y_new < 0) || (x_new >= originalImage.cols) || (y_new >= originalImage.rows))
                    {
                        continue;
                    }

                    if (changed)
                    {
                        //regionsArray[proot_p] = new Region1D(p0, bwImage.rows, bwImage.cols);
                        proot_p = *uf_Find1D(&p0, parentsArray);
                    }

                    // p1 is neighbor of point of interest
                    p1 = p_new;
                    p1root_p = *uf_Find1D(&p1, parentsArray);

                    if ((parentsArray[p1] != UINT_MAX) && (p1root_p != proot_p))
                    {
                        // Entering here means that p1 belongs to some region since has a parent
                        neighbors_tested = true;

                        // Need to union. Three cases: rank1>rank2, rank1<rank2, rank1=rank2
                        point_rank = ranksArray[p0];
                        neighbor_rank = ranksArray[p1root_p];

                        neighborsInRegions = 0;
                        horizontalNeighbors = 0;

                        // old 3*3 loop inline
                        ptemp = p0 - 1;
                        if (*uf_Find1D(&ptemp, parentsArray) == p1root_p)
                        {
                            horizontalNeighbors++;
                            neighborsInRegions++;
                        }

                        ptemp = p0 + 1;
                        if (*uf_Find1D(&ptemp, parentsArray) == p1root_p)
                        {
                            horizontalNeighbors++;
                            neighborsInRegions++;
                        }

                        ptemp = p0 + bwImage.cols;
                        if (*uf_Find1D(&ptemp, parentsArray) == p1root_p)
                        {
                            neighborsInRegions++;
                        }

                        ptemp = p0 - bwImage.cols;
                        if (*uf_Find1D(&ptemp, parentsArray) == p1root_p)
                        {
                            neighborsInRegions++;
                        }

                        // uf_union
                        if (point_rank < neighbor_rank)
                        {
                            parentsArray[proot_p] = p1root_p;
                            if (regionsArray[proot_p] == NULL)
                            {
                                regionsArray[p1root_p]->AttachPoint(proot_p, thresh, bwImage.cols, neighborsInRegions, horizontalNeighbors);
                            }
                            else
                            {
                                regionsArray[p1root_p]->Attach(regionsArray[proot_p], neighborsInRegions, p0 / bwImage.cols, horizontalNeighbors);
                                delete regionsArray[proot_p];
                                regionsArray[proot_p] = NULL;
                            }

                            changed = true;
                        }
                        else if (point_rank > neighbor_rank)
                        {
                            if (regionsArray[proot_p] == NULL)
                            {
                                regionsArray[proot_p] = new Region1D(p0, thresh, bwImage.rows, bwImage.cols);
                            }

                            parentsArray[p1root_p] = proot_p;
                            regionsArray[proot_p]->Attach(regionsArray[p1root_p], neighborsInRegions, p0 / bwImage.cols, horizontalNeighbors);
                            delete regionsArray[p1root_p];
                            regionsArray[p1root_p] = NULL;
                        }
                        else
                        {
                            if (regionsArray[proot_p] == NULL)
                            {
                                regionsArray[proot_p] = new Region1D(p0, thresh, bwImage.rows, bwImage.cols);
                            }

                            parentsArray[p1root_p] = proot_p;
                            ranksArray[proot_p]++;
                            regionsArray[proot_p]->Attach(regionsArray[p1root_p], neighborsInRegions, p0 / bwImage.cols, horizontalNeighbors);
                            delete regionsArray[p1root_p];
                            regionsArray[p1root_p] = NULL;
                        }
                    }
                    else
                    {
                        // Neighbor not in region. Doing nothing
                    }
                }

                // None of neighbors belong to existing region, creating region of 1 point
                if (!neighbors_tested)
                {
                    // Creating region here
                    regionsArray[p0] = new Region1D(p0, thresh, bwImage.rows, bwImage.cols);
                }

                ptemp = *uf_Find1D(&p0, parentsArray);
                regionsArray[ptemp]->CorrectEuler(qtemp);
            }
        }

        t = (double)getTickCount() - t;

        set<Region1D, Region1DComp> retval;

        int regionsCount = 0;
        for(i = 0; i < bwImage.cols * bwImage.rows; i++)
        {
            if (regionsArray[i] != NULL)
            {
                regionsCount++;
                //rectangle(originalImage, regionsArray[i][j]->Bounds(), Scalar(0, 0, 0));

                retval.insert(Region1D(0,
                                       regionsArray[i]->Threshold(),
                                       Rect(regionsArray[i]->Bounds().x - 1, regionsArray[i]->Bounds().y - 1, regionsArray[i]->Bounds().width, regionsArray[i]->Bounds().height),
                                       regionsArray[i]->Area(),
                                       regionsArray[i]->Perimeter(),
                                       regionsArray[i]->Euler(),
                                       regionsArray[i]->AllCrossings(),
                                       bwImage.rows));

/*
                cout << "New region: " << regionsCount << endl;
                cout << "Area: " << regionsArray[i]->Area() << endl;
                cout << "Bounding box " << regionsArray[i]->Bounds().x - 1 << " " << regionsArray[i]->Bounds().y - 1 << " " << regionsArray[i]->Bounds().width << " " << regionsArray[i]->Bounds().height << endl;
                cout << "Perimeter: " << regionsArray[i]->Perimeter() << endl;
                cout << "Euler number: " << regionsArray[i]->Euler() << endl;
                cout << "Crossings: ";
                for(int kk = regionsArray[i]->Bounds().y; kk < regionsArray[i]->Bounds().y + regionsArray[i]->Bounds().height; kk++)
                {
                    cout << regionsArray[i]->Crossings(kk) << " ";
                }
                cout << endl;
                cout << "=====" << endl << endl;
*/
            }
        }

        cout << "Working time: " << t * 1000. / getTickFrequency() << " ms" << endl;

        return retval;
    }

}
