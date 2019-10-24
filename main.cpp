#include <iostream>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <stack>
#include <regex>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <iterator>

using std::string;
using std::vector;
using std::stack;
using namespace std::regex_constants;

class ClosestTimeSolution {
public:
    string nextClosestTime(string time) {
        int curhour = std::stoi(time.substr(0, 2));
        int curmin = curhour * 60 + std::stoi(time.substr(3, 5));

        std::vector<char> times = {time[0], time[1], time[3], time[4]};

        int mindiff = 1000000;
        string curmintime;
        for (char t1 : times) {
            for (char t2 : times) {
                for (char t3 : times) {
                    for (char t4 : times) {
                        int hour = (static_cast<int>(t1) - 48) * 10 + (static_cast<int>(t2) - 48);
                        int min = hour * 60 + (static_cast<int>(t3) - 48) * 10 + (static_cast<int>(t4) - 48);
                        if ((static_cast<int>(t3) - 48) * 10 + (static_cast<int>(t4) - 48) >= 60 || hour > 24) {
                            continue;
                        }
                        if (curmin >= min) {
                            min = min + 24 * 60;
                        }
                        int diff = min - curmin;
                        if (diff < mindiff) {
                            mindiff = diff;
                            curmintime = string({t1, t2, ':', t3, t4});
                        }
                    }
                }
            }
        }
        return curmintime;
    }
};

class LicenseSolution {
public:
    string licenseKeyFormatting(string S, int K) {
        S.erase(std::remove(S.begin(), S.end(), '-'), S.end());
        std::transform(S.begin(), S.end(), S.begin(), ::toupper);

        for (int i = S.length() - K; i > 0; i = i - K) {
            S.insert(i, "-");
        }
        return S;
    }
};

class LongestLengthSolution {
public:
    int lengthLongestPath(string input) {
        input.push_back('\n');

        int lastTabLevel = -1;
        int currentTabLevel = 0;
        std::stack<int> stringStack;
        int currLength = 0;
        int maxLength = 0;
        bool isFile = false;
        for (int index = 0; index < input.size(); ++index) {
            if (input[index] == '\t') {
                currentTabLevel++;
                continue;
            }
            if (input[index] == '.') {
                isFile = true;
            }

            if (input[index] == '\n') {
                if (currentTabLevel < lastTabLevel) {
                    for (int i = 0; i < lastTabLevel - currentTabLevel; ++i) {
                        stringStack.pop();
                    }
                    lastTabLevel = currentTabLevel;
                }

                if (currentTabLevel > lastTabLevel) {
                    int lastLength = stringStack.empty() ? 0 : stringStack.top();
                    stringStack.push(currLength + lastLength + 1);
                    if (stringStack.top() > maxLength && isFile) {
                        maxLength = stringStack.top() - 1;
                    }
                }

                if (currentTabLevel == lastTabLevel) {
                    stringStack.pop();
                    int lastLength = stringStack.empty() ? 0 : stringStack.top();
                    stringStack.push(lastLength + currLength + 1);
                    if (stringStack.top() > maxLength && isFile) {
                        maxLength = stringStack.top() - 1;
                    }
                }


                lastTabLevel = currentTabLevel;
                currentTabLevel = 0;
                currLength = 0;
                isFile = false;
                continue;
            }
            currLength++;
        }
        return maxLength;
    }
};

class BulbSolution {
public:
    int kEmptySlots(vector<int> &bulbs, int K) {
        vector<bool> bulbState(bulbs.size());
        for (int d = 0; d < bulbs.size(); ++d) {
            bulbState[bulbs[d] - 1] = true;

            bool on = false;
            int right = bulbs[d] - 1;
            int left = bulbs[d] - 1;

            int count = 0;
            while (right < bulbState.size()) {
                right = right + 1;
                if (bulbState[right] == false) {
                    count++;
                } else {
                    if (count == K) {
                        return d;
                    }
                    break;
                }
            }

            count = 0;
            while (left > 0) {
                left = left - 1;
                if (bulbState[left] == false) {
                    count++;
                } else {
                    if (count == K) {
                        return d;
                    }
                    break;
                }
            }


        }
        return -1;
    }
};

class EmailSolution {
public:
    int numUniqueEmails(vector<string> &emails) {
        std::unordered_map<string, bool> stringSet;

        for (auto &email : emails) {

            for (int i = 0; i < email.size(); ++i) {
                if (email[i] == '.') {
                    email.erase(i, 1);
                }
                if (email[i] == '@') {
                    break;
                }
            }
        }

        for (auto email : emails) {
            bool isPlus = false;

            int iLocation = -1;
            int ampLocation = -1;
            for (int i = 0; i < email.size(); ++i) {
                if (email[i] == '+' && iLocation == -1) {
                    iLocation = i;
                }

                if (email[i] == '@') {
                    ampLocation = i;
                }
            }
            if (iLocation != -1 && ampLocation != -1) {
                email.erase(iLocation, ampLocation - iLocation);
            }
            stringSet[email] = true;
        }
        return stringSet.size();
    }
};

class FruitSolution {
public:
    int totalFruit(vector<int> &tree) {
        int absMax = 0;
        int currMax = 0;
        int j = 0;
        int fruitTypeA = -1, fruitTypeB = -1;
        int lastFruitAIndex = -1, lastFruitBIndex = -1;
        int lastFruitType = -1;

        while (j < tree.size()) {
            int newFruit = tree[j];

            if (newFruit == fruitTypeA || newFruit == fruitTypeB) {
                lastFruitType = newFruit;
                if (newFruit == fruitTypeB) {
                    lastFruitBIndex = j;
                }

                if (newFruit == fruitTypeA) {
                    lastFruitAIndex = j;
                }

                currMax++;
                if (currMax > absMax) {
                    absMax = currMax;
                }
                j++;
                continue;
            }

            if (newFruit != fruitTypeA && fruitTypeA == -1) {
                lastFruitType = newFruit;
                fruitTypeA = newFruit;
                lastFruitAIndex = j;
                currMax++;
                if (currMax > absMax) {
                    absMax = currMax;
                }
                j++;
                continue;
            }

            if (newFruit != fruitTypeB && fruitTypeB == -1) {
                lastFruitType = newFruit;
                fruitTypeB = newFruit;
                lastFruitBIndex = j;
                currMax++;
                if (currMax > absMax) {
                    absMax = currMax;
                }
                j++;
                continue;
            }

            if (lastFruitType == fruitTypeA) {
                currMax = j - lastFruitBIndex - 1;
                fruitTypeB = -1;
            } else {
                currMax = j - lastFruitAIndex - 1;
                fruitTypeA = -1;
            }
        }
        return absMax;
    }
};

class TilesSolution {
public:
    vector<vector<int>> comb(int N, int K) {
        vector<vector<int>> comblist;

        std::string bitmask(K, 1); // K leading 1's
        bitmask.resize(N, 0); // N-K trailing 0's

        // print integers and permute bitmask
        do {
            std::vector<int> comb;
            for (int i = 0; i < N; ++i) {
                if (bitmask[i]) {
                    comb.push_back(i);
                }
            }
            comblist.push_back(comb);
        } while (std::prev_permutation(bitmask.begin(), bitmask.end()));

        return comblist;
    }

    vector<vector<int>> perm(int N, int K) {
        auto out = comb(N, K);
        vector<vector<int>> evenMore;
        for (auto c : out) {
            do {
                vector<int> cc;
                for (int i = 0; i < c.size(); ++i) {
                    cc.push_back(c[i]);
                }
                evenMore.push_back(cc);
            } while (std::next_permutation(c.begin(), c.end()));
        }
        return evenMore;
    }


    int numTilePossibilities(string tiles) {
        std::unordered_map<string, bool> tilePossibilities;

        for (int i = 1; i < tiles.size() + 1; ++i) {
            vector<vector<int>> c = perm(tiles.size(), i);
            for (auto c1 : c) {
                string s;
                for (auto c2 : c1) {
                    s.push_back(tiles[c2]);
                }
                tilePossibilities[s] = true;
            }
        }
        return tilePossibilities.size();
    }
};

class GardenSolution {
public:
    vector<int> gardenNoAdj(int N, vector<vector<int>> &paths) {
        std::unordered_map<int, std::vector<int>> adj;
        for (auto &p : paths) {
            adj[p[0]].push_back(p[1]);
            adj[p[1]].push_back(p[0]);
        }

        vector<std::unordered_set<int>> remainingGardenPlant(N);
        vector<int> flowerPlanted(N);
        for (auto &garden : remainingGardenPlant) {
            garden = {1, 2, 3, 4};
        }

        for (int i = 0; i < N; ++i) {
            int chose = *remainingGardenPlant[i].begin();
            flowerPlanted[i] = chose;
            for (auto a : adj[i + 1]) {
                remainingGardenPlant[a - 1].erase(chose);
            }
        }

        return flowerPlanted;
    }
};

class MedianSolution {
public:
    double findMedianSortedArrays(vector<int> &nums1, vector<int> &nums2) {
        std::multiset<int> nums3;
        nums3.insert(nums1.begin(), nums1.end());
        nums3.insert(nums2.begin(), nums2.end());
        if (nums3.size() % 2 == 1)
            return *std::next(nums3.begin(), (nums3.size() - 1) / 2);
        else
            return 0.5 *
                   (*std::next(nums3.begin(), (nums3.size()) / 2) + *std::next(nums3.begin(), nums3.size() / 2 - 1));
    }
};

class RotateDigitSolution {
public:
    int rotatedDigits(int N) {
        int Nout = 0;
        for (int i = 1; i <= N; ++i) {
            vector<int> digits;
            int num = i;
            while (num != 0) {
                digits.push_back(num % 10);
                num = num / 10;
            }
            bool bad2 = true;
            for (auto d : digits) {
                if (d == 3 || d == 4 || d == 7) {
                    bad2 = false;
                    break;
                }
            }
            if (!bad2) {
                continue;
            }

            if (digits.front() == 0) {
                continue;
            }

            bool bad = true;

            for (int k = 0; k < digits.size(); ++k) {
                int left = digits[k];
                int right = digits[digits.size() - k - 1];
                if ((left == 1 && right == 1) || (left == 8 && right == 8) || (left == 0 && right == 0) ||
                    (left == 2 && right == 5) || (left == 9 && right == 6) || (left == 5 && right == 2) ||
                    (left == 6 && right == 9)) {
                    continue;
                }
                bad = false;
                break;
            }
            if (bad) {
                continue;
            }

            Nout++;
        }
        return Nout;
    }
};


struct ListNode {
    int val;
    ListNode *next;

    ListNode(int x) : val(x), next(NULL) {}
};

class Solution {
public:
    ListNode *addTwoNumbers(ListNode *l1, ListNode *l2) {
        while (l1->next != nullptr) {
            l1++;
            l2++;
            l2->val += l1->val;
        }
        return l2;
    }
};

class DominoSolution {
public:
    int minDominoRotations(vector<int> &A, vector<int> &B) {
        int mintop = 1000000;

        for (int i = 1; i <= 6; ++i) {
            int topval = 0;
            int botval = 0;
            bool fail = false;
            for (int j = 0; j < A.size(); ++j) {
                if (A[j] != i && B[j] != i) {
                    fail = true;
                    break;
                }

                if (A[j] == i && B[j] == i) {
                    continue;
                }

                if (A[j] == i && B[j] != i) {
                    topval++;
                } else {
                    botval++;
                }
            }
            if (fail) continue;

            int minturns = topval > botval ? botval : topval;
            if (minturns < mintop) {
                mintop = minturns;
            }
        }
        return mintop == 1000000 ? -1 : mintop;
    }
};

class GraphTreeSolution {
public:
    bool validTree(int n, vector<vector<int>> &edges) {
        if (edges.empty()) {
            return n == 1;
        }

        std::unordered_map<int, std::unordered_set<int>> adj_list;
        for (auto edge : edges) {
            adj_list[edge[0]].insert(edge[1]);
            adj_list[edge[1]].insert(edge[0]);
        }

        stack<int> stack;

        // Find vertex with only outgoing edges
        stack.push(edges[0][0]);

        std::unordered_set<int> visited;
        while(!stack.empty()) {
            int n = stack.top();
            stack.pop();
            if (visited.find(n) != visited.end()) {
                return false;
            }
            visited.insert(n);
            for (auto adj : adj_list[n]) {
                if (visited.find(adj) != visited.end() && adj_list[adj].find(n) != adj_list[adj].end()) {
                    continue;
                }
                stack.push(adj);
            }
        }

        return visited.size() == n;
    }
};

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class TreeSolution {
    vector<TreeNode*> heads;

public:
    vector<TreeNode*> delNodes(TreeNode* root, vector<int>& to_delete) {
        delNodesHelper(root, nullptr, to_delete, true);
        return heads;
    }

    void delNodesHelper(TreeNode* root, TreeNode* parent, vector<int>& to_delete, bool parent_deleted) {
        if (root == nullptr) {
            return;
        }

        auto left = root->left;
        auto right = root->right;
        for(auto val : to_delete) {
            if (root->val == val) {
                if (parent) {
                    if (parent->left == root) {
                        parent->left = nullptr;
                    } else {
                        parent->right = nullptr;
                    }
                }

                root = nullptr;
                break;
            }
        }

        if (parent_deleted && root != nullptr) {
            heads.push_back(root);
        }

        delNodesHelper(left, root, to_delete, root == nullptr);
        delNodesHelper(right, root, to_delete, root == nullptr);
    }
};

class KnightSolution {
public:
//    double knightProbability(int N, int K, int r, int c) {
//        if (r < 0 || r >= N || c < 0 || c >= N) {
//            return 0;
//        }
//
//        if (K == 0) {
//            return 1;
//        }
//
//        double totalKnightProbability = 0.0f;
//        totalKnightProbability += 0.125 * knightProbability(N, K-1, r + 2, c + 1);
//        totalKnightProbability += 0.125 * knightProbability(N, K-1, r + 2, c - 1);
//        totalKnightProbability += 0.125 * knightProbability(N, K-1, r + 1, c + 2);
//        totalKnightProbability += 0.125 * knightProbability(N, K-1, r + 1, c - 2);
//        totalKnightProbability += 0.125 * knightProbability(N, K-1, r - 2, c + 1);
//        totalKnightProbability += 0.125 * knightProbability(N, K-1, r - 2, c - 1);
//        totalKnightProbability += 0.125 * knightProbability(N, K-1, r - 1, c + 2);
//        totalKnightProbability += 0.125 * knightProbability(N, K-1, r - 1, c - 2);
//
//        return totalKnightProbability;
//    }

    bool onGrid(int N, int r, int c) {
        if (r < 0 || r >= N || c < 0 || c >= N) {
            return false;
        }
        return true;
    }

    double knightProbability(int N, int K, int r, int c) {
        float grid[N][N];
        float grid_old[N][N];

        if (K == 0) {
            return 1.0;
        }

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                grid[i][j] = 0;
                grid_old[i][j] = 1;
            }
        }

        for (int i = 0; i < K; ++i) {
            for (int a = 0; a < N; ++a) {
                for (int b = 0; b < N; ++b) {
                    grid[a][b] = 0.0;
                    grid[a][b] += 0.125 * (onGrid(N, a + 2, b + 1) ? grid_old[a + 2][b + 1] : 0.0f);
                    grid[a][b] += 0.125 * (onGrid(N, a + 2, b - 1) ? grid_old[a + 2][b - 1] : 0.0f);
                    grid[a][b] += 0.125 * (onGrid(N, a + 1, b + 2) ? grid_old[a + 1][b + 2] : 0.0f);
                    grid[a][b] += 0.125 * (onGrid(N, a + 1, b - 2) ? grid_old[a + 1][b - 2] : 0.0f);
                    grid[a][b] += 0.125 * (onGrid(N, a - 2, b + 1) ? grid_old[a - 2][b + 1] : 0.0f);
                    grid[a][b] += 0.125 * (onGrid(N, a - 2, b - 1) ? grid_old[a - 2][b - 1] : 0.0f);
                    grid[a][b] += 0.125 * (onGrid(N, a - 1, b + 2) ? grid_old[a - 1][b + 2] : 0.0f);
                    grid[a][b] += 0.125 * (onGrid(N, a - 1, b - 2) ? grid_old[a - 1][b - 2] : 0.0f);
                }
            }

            for (int a = 0; a < N; ++a) {
                for (int b = 0; b < N; ++b) {
                    grid_old[a][b] = grid[a][b];
                }
            }
        }

        return grid[r][c];
    }
};

class BadVersionSolution {
public:
    bool firstBadVersionHelper(int min, int max) {
        if (min == max) {
            return false;
        }

        int current = (max - min) / 2;
        std::cout << current << std::endl;
        firstBadVersionHelper(min, max/2);
        firstBadVersionHelper(max/2 + 1, max);

        return true;
    }

    int firstBadVersion(int n) {
        firstBadVersionHelper(0, n);
    }
};

class SnapshotArray {
    int snap_num = 0;
    std::unordered_map<int,std::map<int, int>> snaps;
public:
    SnapshotArray(int length) {}

    void set(int index, int val) {
        snaps[index][snap_num] = val;
    }

    int snap() {
        return snap_num++;
    }

    int get(int index, int snap_id) {
        int current_snap = 0;
        for (std::pair<int, int> snap : snaps[index]) {
            if (snap_id >= snap.first) {
                current_snap = snap.second;
            } else {
                break;
            }
        }
        return  current_snap;
    }
};

class EvaluateDivisionSolution {
public:
    vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
        std::unordered_map<string, std::unordered_map<string, float>> adj_list;
        std::unordered_set<string> nodes;
        for (int i = 0; i < equations.size(); ++i) {
            adj_list[equations[i][0]][equations[i][1]] = values[i];
            adj_list[equations[i][1]][equations[i][0]] = 1.0f/values[i];
            nodes.insert(equations[i][0]);
            nodes.insert(equations[i][1]);
        }

        vector<double> output;
        for (auto q : queries) {

            if (nodes.find(q[0]) == nodes.end()) {
                output.push_back(-1);
                continue;
            }

            // Do a DFS
            stack<string> node;
            std::unordered_set<string> visited;
            std::unordered_map<string, float> ratio;

            node.push(q[0]);
            ratio[q[0]] = 1;

            while(!node.empty()) {
                string next = node.top();
                node.pop();
                visited.insert(next);

                if (next == q[1]) {
                    break;
                }

                for (const auto& adj : adj_list[next]) {
                    if (visited.find(adj.first) != visited.end()) {
                        continue;
                    }

                    ratio[adj.first] = ratio[next] * adj.second;
                    node.push(adj.first);
                }
            }

            output.push_back(ratio.find(q[1]) != ratio.end() ? ratio[q[1]] : -1);
        }
        return output;
    }
};

class StoneSolution {
public:
    int removeStones(vector<vector<int>>& stones) {

    }
};

class PalidromeSolution {
public:
    string longestPalindrome(string s) {
        if (s.empty() || s.length() < 1) return "";
        int start = 0, end = 0;
        for (int i = 0; i < s.length(); i++) {
            int len1 = expandAroundCenter(s, i, i);
            int len2 = expandAroundCenter(s, i, i + 1);
            int len = std::max(len1, len2);
            if (len > end - start) {
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }
        }
        return s.substr(start, end - start + 1);
    }

    int expandAroundCenter(string s, int left, int right) {
        int L = left, R = right;
        while (L >= 0 && R < s.length() && s[L] == s[R]) {
            L--;
            R++;
        }
        return R - L - 1;
    }
};

class MaxSubarraySolution {
public:
    int maxSubArray(vector<int>& nums) {
        int curr_sum = nums[0];
        int max_sum = nums[0];

        for (int i = 0; i < nums.size(); ++i) {
            curr_sum = std::max(nums[i], curr_sum + nums[i]);
            max_sum = std::max(max_sum, curr_sum);
        }

        return max_sum;
    }
};

class BuySellStockSolution {
public:
    int maxProfit(std::vector<int> prices) {
        int minprice = 1000000;
        int maxprofit = 0;
        for (int i = 0; i < prices.size(); i++) {
            if (prices[i] < minprice)
                minprice = prices[i];
            else if (prices[i] - minprice > maxprofit)
                maxprofit = prices[i] - minprice;
        }
        return maxprofit;
    }
};

int main() {
    BuySellStockSolution ss;
    vector<int> nums = {7,1,5,3,6,4};
    auto s = ss.maxProfit(nums);
    std::cout << s << std::endl;
    return 0;
}