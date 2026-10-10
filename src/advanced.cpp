#include "advanced.h"
#include <QJsonArray>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pad {
namespace {
void checkCancel(const std::function<bool()>& cancelled) {
    if (cancelled && cancelled()) throw std::runtime_error("Cancelled");
}
float bilinear(const cv::Mat& field, const cv::Point2f& p) {
    const int x = int(std::floor(p.x)), y = int(std::floor(p.y));
    const float dx = p.x-x, dy = p.y-y;
    return (1-dy)*((1-dx)*field.at<float>(y,x)+dx*field.at<float>(y,x+1))
          +dy*((1-dx)*field.at<float>(y+1,x)+dx*field.at<float>(y+1,x+1));
}
bool inside(const cv::Point2f& p, cv::Size size) {
    return p.x >= 1 && p.y >= 1 && p.x < size.width-2 && p.y < size.height-2;
}
}

cv::Mat correctIllumination(const cv::Mat& gray, double sigma, bool divide) {
    cv::Mat source, background, corrected;
    gray.convertTo(source, CV_32F);
    cv::GaussianBlur(source, background, cv::Size(), sigma, sigma, cv::BORDER_REFLECT101);
    if (divide) {
        cv::max(background, 1, background);
        cv::divide(source, background, corrected, cv::mean(source)[0]);
    } else {
        corrected = source-background+128;
    }
    corrected.convertTo(corrected, CV_8U);
    return corrected;
}

cv::Mat guidedSmooth(const cv::Mat& gray, int radius, double epsilon) {
    // Self-guided filter: epsilon is in normalized [0,1] intensity squared.
    cv::Mat source, mean, meanSquared, variance, a, b, meanA, meanB, result;
    gray.convertTo(source, CV_32F, 1.0/255);
    const cv::Size kernel(2*radius+1, 2*radius+1);
    cv::boxFilter(source, mean, -1, kernel, cv::Point(-1,-1), true, cv::BORDER_REFLECT101);
    cv::boxFilter(source.mul(source), meanSquared, -1, kernel, cv::Point(-1,-1), true, cv::BORDER_REFLECT101);
    variance = meanSquared-mean.mul(mean);
    cv::max(variance, 0, variance);
    cv::divide(variance, variance+epsilon, a);
    b = mean-a.mul(mean);
    cv::boxFilter(a, meanA, -1, kernel, cv::Point(-1,-1), true, cv::BORDER_REFLECT101);
    cv::boxFilter(b, meanB, -1, kernel, cv::Point(-1,-1), true, cv::BORDER_REFLECT101);
    result = meanA.mul(source)+meanB;
    result.convertTo(result, CV_8U, 255);
    return result;
}

cv::Mat sauvolaMask(const cv::Mat& gray, int block, double k, double range, bool inverse) {
    cv::Mat source, mean, squared, deviation, threshold, result;
    gray.convertTo(source, CV_32F);
    const cv::Size kernel(block,block);
    cv::boxFilter(source, mean, -1, kernel, cv::Point(-1,-1), true, cv::BORDER_REFLECT101);
    cv::boxFilter(source.mul(source), squared, -1, kernel, cv::Point(-1,-1), true, cv::BORDER_REFLECT101);
    deviation = squared-mean.mul(mean);
    cv::max(deviation, 0, deviation);
    cv::sqrt(deviation, deviation);
    threshold = mean.mul(1+k*(deviation/range-1));
    cv::compare(source, threshold, result, inverse ? cv::CMP_LE : cv::CMP_GT);
    return result;
}

cv::Mat watershedSplit(const cv::Mat& mask, const cv::Mat& original, int radius, double fraction) {
    if (cv::countNonZero(mask)==0) return mask.clone();
    // Pad to keep objects touching an image edge away from watershed's forced border.
    cv::Mat padded, color;
    cv::copyMakeBorder(mask,padded,1,1,1,1,cv::BORDER_CONSTANT,0);
    cv::copyMakeBorder(original,color,1,1,1,1,cv::BORDER_REPLICATE);
    cv::Mat distance, maximum, peaks, components;
    cv::distanceTransform(padded,distance,cv::DIST_L2,cv::DIST_MASK_PRECISE);
    cv::dilate(distance,maximum,cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(2*radius+1,2*radius+1)));
    const int regions=cv::connectedComponents(padded,components,8);
    std::vector<float> maxDistance(regions,0);
    for(int y=0;y<padded.rows;y++) for(int x=0;x<padded.cols;x++)
        maxDistance[components.at<int>(y,x)]=std::max(maxDistance[components.at<int>(y,x)],distance.at<float>(y,x));
    peaks=cv::Mat::zeros(padded.size(),CV_8U);
    for(int y=0;y<padded.rows;y++) for(int x=0;x<padded.cols;x++) {
        const int label=components.at<int>(y,x);
        const float v=distance.at<float>(y,x);
        if(label && v>=maximum.at<float>(y,x)-1e-5f && v>=maxDistance[label]*fraction) peaks.at<uchar>(y,x)=255;
    }
    cv::Mat markers;
    cv::connectedComponents(peaks,markers,8);
    markers+=1;
    markers.setTo(0,(padded>0)&(peaks==0));
    cv::watershed(color,markers);
    cv::Mat result=cv::Mat::zeros(padded.size(),CV_8U);
    result.setTo(255,(markers>1)&(padded>0));
    return result(cv::Rect(1,1,mask.cols,mask.rows)).clone();
}

std::vector<EdgeContour> localizeEdges(const cv::Mat& mask, const cv::Mat& original,
                                     int radius, double sigma, double minGradient,
                                     int span, const std::function<bool()>& cancelled) {
    cv::Mat gray, source, gx, gy;
    if(original.channels()==1)gray=original;else cv::cvtColor(original,gray,cv::COLOR_BGR2GRAY);
    gray.convertTo(source,CV_32F);
    if(sigma>0) cv::GaussianBlur(source,source,cv::Size(),sigma,sigma,cv::BORDER_REFLECT101);
    cv::Sobel(source,gx,CV_32F,1,0,3,1.0/8,0,cv::BORDER_REFLECT101);
    cv::Sobel(source,gy,CV_32F,0,1,3,1.0/8,0,cv::BORDER_REFLECT101);
    std::vector<std::vector<cv::Point>> seeds;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask,seeds,hierarchy,cv::RETR_TREE,cv::CHAIN_APPROX_NONE);
    size_t count=0;
    for(const auto& contour:seeds) count+=contour.size();
    if(count>500000) throw std::runtime_error("More than 500,000 boundary points; filter mask noise first");
    std::vector<EdgeContour> contours;
    for(size_t c=0;c<seeds.size();c++) {
        checkCancel(cancelled);
        EdgeContour out;out.parent=hierarchy[c][3];
        int depth=0;for(int parent=out.parent;parent>=0;parent=hierarchy[parent][3])depth++;
        out.hole=depth%2;
        const auto& seed=seeds[c];const int n=int(seed.size());
        for(int i=0;i<n;i++) {
            if(i%1024==0)checkCancel(cancelled);
            EdgePoint point;point.seed=point.position=cv::Point2f(seed[i]);
            if(n<5){out.points.push_back(point);continue;}
            const int reach=std::min(span,(n-1)/2);
            const cv::Point2f tangent=cv::Point2f(seed[(i+reach)%n]-seed[(i-reach+n)%n]);
            const double norm=cv::norm(tangent);
            if(norm<1e-6){out.points.push_back(point);continue;}
            point.normal=cv::Point2f(float(-tangent.y/norm),float(tangent.x/norm));
            std::vector<float> response(2*radius+1,-1);
            int best=radius;float peak=-1;
            for(int t=-radius;t<=radius;t++) {
                const auto sample=point.seed+point.normal*float(t);
                if(!inside(sample,mask.size()))continue;
                const float strength=std::abs(bilinear(gx,sample)*point.normal.x+bilinear(gy,sample)*point.normal.y);
                response[t+radius]=strength;
                // Prefer the closest peak when equal; prevents arbitrary movement across a plateau.
                if(strength>peak+1e-5f || (std::abs(strength-peak)<=1e-5f && std::abs(t)<std::abs(best-radius))){peak=strength;best=t+radius;}
            }
            point.strength=std::max(peak,0.f);
            if(peak>=minGradient && best>0 && best<2*radius && response[best-1]>=0 && response[best+1]>=0) {
                point.position=point.seed+point.normal*float(best-radius);
                point.before=response[best-1];point.peak=peak;point.after=response[best+1];point.localized=true;
            }
            out.points.push_back(point);
        }
        contours.push_back(std::move(out));
    }
    return contours;
}

void refineSubpixel(std::vector<EdgeContour>& contours,double minCurvature,
                    const std::function<bool()>& cancelled) {
    for(auto& contour:contours) {
        checkCancel(cancelled);
        for(auto& point:contour.points) {
            if(!point.localized)continue;
            const double curvature=2*point.peak-point.before-point.after;
            if(curvature<=minCurvature)continue;
            // Vertex of the quadratic through samples at -1,0,+1 along the normal.
            const double offset=(point.after-point.before)/(2*curvature);
            if(!std::isfinite(offset)||std::abs(offset)>0.50001)continue;
            point.position+=point.normal*float(std::clamp(offset,-0.5,0.5));
            point.refined=true;
        }
    }
}

cv::Mat boundaryOverlay(const cv::Mat& image,const std::vector<EdgeContour>& contours) {
    cv::Mat out;if(image.channels()==1)cv::cvtColor(image,out,cv::COLOR_GRAY2BGR);else out=image.clone();
    for(const auto& contour:contours) {
        std::vector<cv::Point> points;
        for(const auto& p:contour.points)points.emplace_back(cvRound(p.position.x*256),cvRound(p.position.y*256));
        if(points.size()>1)cv::polylines(out,points,true,cv::Scalar(255,230,0),1,cv::LINE_AA,8);
        for(const auto& p:contour.points)if(!p.localized)cv::circle(out,cv::Point(cvRound(p.position.x),cvRound(p.position.y)),1,cv::Scalar(0,130,255),-1);
    }
    return out;
}

QJsonObject boundariesJson(const Result& result) {
    QJsonArray contours;int localized=0,refined=0,total=0;
    for(const auto& contour:result.boundaries) {
        QJsonArray points;std::vector<cv::Point2f> positions;
        for(const auto& p:contour.points) {
            points.append(QJsonObject{{"x",p.position.x},{"y",p.position.y},{"seed_x",p.seed.x},{"seed_y",p.seed.y},
                                     {"gradient",p.strength},{"localized",p.localized},{"refined",p.refined}});
            positions.push_back(p.position);total++;localized+=p.localized;refined+=p.refined;
        }
        QJsonObject item{{"parent",contour.parent},{"hole",contour.hole},{"points",points}};
        if(positions.size()>=3) {
            const auto rect=cv::minAreaRect(positions);cv::Point2f corners[4];rect.points(corners);QJsonArray vertices;
            for(const auto& p:corners)vertices.append(QJsonArray{p.x,p.y});
            item["min_area_rectangle"]=QJsonObject{{"corners",vertices},{"width",rect.size.width},{"height",rect.size.height},{"angle_degrees",rect.angle}};
        }
        contours.append(item);
    }
    return {{"schema_version",1},{"enabled",result.hasBoundaries},{"subpixel_requested",result.subpixel},
            {"coordinate_system","original image; x=column,y=row; integer coordinates are pixel centers; units=pixel"},
            {"method","Sobel gradient normal-profile peak; optional 3-sample quadratic interpolation"},
            {"mask_policy","measurement only; binary mask and integer detection boxes unchanged"},
            {"points",total},{"localized",localized},{"refined",refined},{"contours",contours}};
}
}
