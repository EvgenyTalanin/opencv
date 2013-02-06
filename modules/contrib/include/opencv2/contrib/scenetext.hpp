#include <iostream>
#define SMALL_SIZE 16
#define SMALL_SIZE_MIDDLE 8

using namespace std;

namespace cv
{
class Region
{
private:
    Point start;
    Rect bounds;
    int area;
    int perimeter;
    int euler;
    int imageh;
    int* crossings;
public:
    Region();
    Region(Point, int);
    Region(Point, Rect, int, int, int, int*, int);
    ~Region();
    void Attach(Region*, int, int, int);

    Point Start();
    void CorrectEuler(int);
    Rect Bounds();
    int Area();
    int Perimeter();
    int Euler();
    int Crossings(int);
    int* AllCrossings();
    int BoundsArea();
};

class Region1D
{
private:
    unsigned start;
    unsigned thresh;
    Rect bounds;
    int area;
    int perimeter;
    int euler;
    int imageh;
    int* crossings;
    int top_of_small;
    int crossings_small[SMALL_SIZE];
public:
    ~Region1D();
    Region1D(unsigned, unsigned, int, int);
    Region1D(unsigned, unsigned, Rect, int, int, int, int*, int);
    void Attach(Region1D*, int, int, int);
    void AttachPoint(unsigned, unsigned, int, int, int);

    unsigned Start();
    unsigned Threshold();
    Rect Bounds();
    int Area();
    int Perimeter();
    int Euler();
    int BoundsArea();

    void CorrectEuler(int);

    int TopOfSmall();
    int* AllCrossings();
    inline int Crossings(int _y)
    {
        if (crossings == NULL)
        {
            if ((_y >= bounds.y) && (_y <= bounds.y + bounds.height - 1))
            {
                return crossings_small[_y - top_of_small];
            }
            else
            {
                return 0;
            }
        }
        else
        {
            return crossings[_y];
        }
    }
};

struct RegionComp
{
    bool operator() (Region _one, Region _other)
    {
        if (_one.Bounds().x < _other.Bounds().x) return true;
        if (_one.Bounds().x > _other.Bounds().x) return false;
        if (_one.Bounds().y < _other.Bounds().y) return true;
        if (_one.Bounds().y > _other.Bounds().y) return false;
        if (_one.Area() < _other.Area()) return true;
        if (_one.Area() > _other.Area()) return false;
        if (_one.Perimeter() < _other.Perimeter()) return true;
        if (_one.Perimeter() > _other.Perimeter()) return false;
        return false;
    }
};

struct Region1DComp
{
    bool operator() (Region1D _one, Region1D _other)
    {
        if (_one.Bounds().x < _other.Bounds().x) return true;
        if (_one.Bounds().x > _other.Bounds().x) return false;
        if (_one.Bounds().y < _other.Bounds().y) return true;
        if (_one.Bounds().y > _other.Bounds().y) return false;
        if (_one.Area() < _other.Area()) return true;
        if (_one.Area() > _other.Area()) return false;
        if (_one.Perimeter() < _other.Perimeter()) return true;
        if (_one.Perimeter() > _other.Perimeter()) return false;
        return false;
    }
};

class SceneTextLocalizer
{
private:
    Mat _originalImage;
    int threshValue;
    Point* uf_Find(Point*, Point**);
    inline unsigned* uf_Find1D(unsigned*, unsigned*);
    inline bool uf_CheckStub(unsigned*, unsigned*);
public:
    SceneTextLocalizer();
    SceneTextLocalizer(Mat, int);
    set<Region, RegionComp> GroundTruth();
    set<Region1D, Region1DComp> MatasLike();
};

}
