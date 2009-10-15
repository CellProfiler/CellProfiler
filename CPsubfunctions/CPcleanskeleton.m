function real_skel = CPcleanskeleton(skel)

%This function takes a skeleton created by bwmorph(I,'skel',Inf) and
%removes all branchpoints that will not break up the skeleton(s).
%The output skeleton will not contain multiple branchpoints at the same
%intersection.
% Originally written by C. Wahlby


[all_skel num_skel]=bwlabel(skel);
real_skel=zeros(size(skel));
for i=1:num_skel
%     if rem(num_skel-i,5) == 0
%         disp(num_skel - i)
%     end
    in_skel=zeros(size(skel));
    in_skel(all_skel==i)=1;
    eul=bweuler(in_skel);
    k=1;
    change=1;
    while(change)
        change=0;
        all_bp=bwmorph(in_skel,'branchpoints');
        f=find(all_bp);
        for n=length(f):-1:1
            in_skel(f(n))=0;
            [l num_obj]=bwlabel(in_skel);
            if num_obj>1 || bweuler(in_skel,8)<eul%to prevent new holes at orthogonal intersections
                in_skel(f(n))=1;
                k=k+1;
            else
                change=1;
            end
        end
    end
    real_skel=real_skel+in_skel;
end