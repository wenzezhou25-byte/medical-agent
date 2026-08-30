# -*- coding: utf-8 -*-
"""高德地图：地址解析、POI 检索、路线规划与附近医院推荐。

从 app.py 拆出。不依赖 Streamlit。
"""
import requests
import traceback
from config import GAODE_MAP_KEY

# 附近医院检索半径（米）：应用层与默认参数统一使用同一常量，避免两侧口径不一致。
NEARBY_HOSPITAL_RADIUS = 10000


def get_route_info(origin_lat, origin_lon, dest_lat, dest_lon, api_key):
    base_url = "https://restapi.amap.com/v3/direction"
    drive_res, walk_res = "🚗 --", "🚶 --"
    drive_distance_m, walk_distance_m = None, None
    try:
        d_params = {"origin": f"{origin_lon},{origin_lat}", "destination": f"{dest_lon},{dest_lat}", "key": api_key,
                    "extensions": "base", "output": "json", "strategy": 2}
        resp = requests.get(f"{base_url}/driving", params=d_params, timeout=8)
        data = resp.json()
        if data.get("status") == "1" and data.get("route", {}).get("paths"):
            path = data["route"]["paths"][0]
            drive_distance_m = int(path["distance"])
            drive_res = f"🚗 {round(int(path['duration']) / 60)}分 ({round(int(path['distance']) / 1000, 1)}km)"
        else:
            # 高德 API 返回失败时保留 info 以便排查（如 USERKEY_PLAT_NOMATCH / OUT_OF_SERVICE）
            info = data.get("info") or "未知错误"
            drive_res = f"🚗 不可达 ({info})"
            print(f"[get_route_info] driving 接口失败 status={data.get('status')} info={info}")
    except Exception:
        print("[get_route_info] driving 请求异常")
        print(traceback.format_exc())
    try:
        w_params = {"origin": f"{origin_lon},{origin_lat}", "destination": f"{dest_lon},{dest_lat}", "key": api_key,
                    "extensions": "base", "output": "json"}
        resp = requests.get(f"{base_url}/walking", params=w_params, timeout=8)
        data = resp.json()
        if data.get("status") == "1" and data.get("route", {}).get("paths"):
            path = data["route"]["paths"][0]
            walk_distance_m = int(path["distance"])
            walk_res = f"🚶 {round(int(path['duration']) / 60)}分 ({round(int(path['distance']) / 1000, 1)}km)"
        else:
            info = data.get("info") or "未知错误"
            walk_res = f"🚶 不可达 ({info})"
            print(f"[get_route_info] walking 接口失败 status={data.get('status')} info={info}")
    except Exception:
        print("[get_route_info] walking 请求异常")
        print(traceback.format_exc())
    # 为了与高德地图默认“驾车路线”一致，排序只采用驾车距离
    route_distance_m = drive_distance_m
    return drive_res, walk_res, route_distance_m


def geocode_address(address, api_key):
    if not address:
        return None, None
    address = str(address).strip()

    # 先按 POI 文本检索定位，避免“学校/园区”等关键词被 geocode 解析到偏移点
    try:
        poi_resp = requests.get(
            "https://restapi.amap.com/v3/place/text",
            params={"keywords": address, "key": api_key, "output": "json", "offset": 1},
            timeout=5,
        )
        poi_data = poi_resp.json()
        if poi_data.get("status") == "1" and poi_data.get("pois"):
            loc = poi_data["pois"][0].get("location")
            if loc:
                lon, lat = loc.split(",")
                return float(lat), float(lon)
        else:
            print(f"[geocode_address] POI 解析失败 address={address} status={poi_data.get('status')} info={poi_data.get('info')}")
    except Exception:
        print(f"[geocode_address] POI 请求异常 address={address}")
        print(traceback.format_exc())

    url = "https://restapi.amap.com/v3/geocode/geo"
    params = {"address": address, "key": api_key, "output": "json"}
    try:
        resp = requests.get(url, params=params, timeout=3)
        data = resp.json()
        if data.get("status") == "1" and data.get("geocodes"):
            loc = data["geocodes"][0]["location"]
            lon, lat = loc.split(",")
            return float(lat), float(lon)
        else:
            print(f"[geocode_address] geocode 接口失败 address={address} status={data.get('status')} info={data.get('info')}")
    except Exception:
        print(f"[geocode_address] geocode 请求异常 address={address}")
        print(traceback.format_exc())
    return None, None


def search_poi_candidates(keyword, api_key, limit=8):
    if not keyword:
        return []
    try:
        resp = requests.get(
            "https://restapi.amap.com/v3/place/text",
            params={"keywords": str(keyword).strip(), "key": api_key, "output": "json", "offset": limit},
            timeout=5,
        )
        data = resp.json()
        if data.get("status") != "1" or not data.get("pois"):
            if data.get("status") != "1":
                print(f"[search_poi_candidates] 接口失败 keyword={keyword} status={data.get('status')} info={data.get('info')}")
            return []
        candidates = []
        for poi in data.get("pois", [])[:limit]:
            loc = poi.get("location")
            if not loc:
                continue
            candidates.append({
                "name": poi.get("name", ""),
                "address": poi.get("address", ""),
                "location": loc,
                "id": poi.get("id", ""),
            })
        return candidates
    except Exception:
        print(f"[search_poi_candidates] 请求异常 keyword={keyword}")
        print(traceback.format_exc())
        return []


# 医疗机构类型评分：数字越大优先级越高。
# 综合医院/专科医院 > 急救中心/疾控 > 卫生服务中心 > 卫生院 > 门诊部 > 诊所
# 放在 search_nearby_hospitals 之前，便于在召回阶段提前停止并预填 _type_score 字段。
MEDICAL_TYPE_SCORE_RULES = [
    (5, ["医院"]),
    (4, ["急救中心", "急救站", "疾控", "疾病预防"]),
    (3, ["社区卫生服务中心", "社区卫生服务站", "卫生服务中心", "卫生服务站"]),
    (2, ["卫生院", "乡镇卫生院"]),
    (1, ["门诊部", "门诊", "医务室", "护理院", "护理站", "诊所"]),
]


def score_medical_institution_type(name):
    """根据机构名称给出类型评分；未命中给 0（仍参与排序，不硬过滤）。"""
    if not name:
        return 0
    name_str = str(name)
    best = 0
    for score, keywords in MEDICAL_TYPE_SCORE_RULES:
        if any(kw in name_str for kw in keywords):
            best = max(best, score)
    return best


def search_nearby_hospitals(location, radius=NEARBY_HOSPITAL_RADIUS):
    if not GAODE_MAP_KEY: return [
        {"name": "⚠️ 未配置地图 API", "address": "", "distance": "-", "tel": "-", "location": ""}]
    try:
        query_lat, query_lon = geocode_address(location, GAODE_MAP_KEY)
        if query_lat is None or query_lon is None:
            return [{"name": "❌ 地址解析失败", "address": "", "distance": "-", "tel": "-", "location": ""}]
        location = f"{query_lon},{query_lat}"

        # 扩大 POI 召回：分别查询多个关键词并合并结果。
        # 不再只查“医院”，避免卫生院/社区服务中心/门诊/诊所/急救/疾控等在 API 阶段就被漏掉。
        # 关键词列表控制在 8 个以内，避免请求过多拖慢 UI。
        keywords = [
            "医院",
            "卫生院",
            "社区卫生服务中心",
            "社区卫生服务站",
            "门诊部",
            "诊所",
            "急救中心",
            "疾控中心",
        ]

        blacklist = ["酒店", "宾馆", "餐厅", "超市", "学校", "公司"]
        # whitelist 覆盖各类正规医疗机构名称特征，不因名字不含“医院”就丢弃
        whitelist = ["医院", "卫生", "诊所", "门诊", "疾控", "急救", "医务", "护理", "社区卫生服务"]

        def _parse_distance(d_str):
            try:
                d_str = str(d_str).strip()
                if '公里' in d_str:
                    return float(d_str.replace('公里', '')) * 1000
                elif '米' in d_str:
                    return float(d_str.replace('米', ''))
                else:
                    return float(d_str)
            except Exception:
                return 999999

        seen = set()  # 去重 key 集合：优先 POI id，否则 name+location
        hospitals = []
        soft_candidates = []  # fallback 储备：未命中 whitelist 但通过黑名单的 POI
        any_keyword_succeeded = False  # 至少一个关键词请求成功
        # 轻量提前停止阈值：
        # - 候选总数达到 50 即停止后续关键词请求（与最终返回上限一致）；
        # - 已收集到 20 个“高质量”机构（类型评分 >= 3，即医院/急救疾控/社区卫生服务中心）
        #   也停止，避免在已经足够好时继续打 5 秒超时的网络请求。
        HARD_COUNT_LIMIT = 50
        HIGH_QUALITY_THRESHOLD = 20
        high_quality_count = 0

        for kw in keywords:
            try:
                search_resp = requests.get("https://restapi.amap.com/v3/place/around", params={
                    "location": location, "keywords": kw,
                    "radius": radius, "key": GAODE_MAP_KEY, "output": "json", "offset": 50
                }, timeout=5)
                search_data = search_resp.json()
                if search_data.get("status") != "1":
                    info = search_data.get("info") or "未知错误"
                    print(f"[search_nearby_hospitals] keyword={kw} 接口失败 status={search_data.get('status')} info={info}")
                    continue
                any_keyword_succeeded = True
                pois = search_data.get("pois") or []
                for poi in pois:
                    name = poi.get("name", "")
                    if not name:
                        continue
                    if any(word in name for word in blacklist):
                        continue
                    poi_location = poi.get("location", "")
                    if not poi_location:
                        continue
                    # 去重：有 id 用 id，否则用 name+location
                    poi_id = poi.get("id", "")
                    dedupe_key = f"id:{poi_id}" if poi_id else f"nl:{name}|{poi_location}"
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)

                    # 预计算类型评分，避免 tab2 重复计算；soft_candidates 也保留以备 fallback 后再评分
                    type_score = score_medical_institution_type(name)
                    item = {
                        "name": name,
                        "address": poi.get("address", ""),
                        "distance": poi.get("distance", ""),
                        "tel": poi.get("tel", ""),
                        "location": poi_location,
                        "_matched_keyword": kw,  # 仅用于调试，不影响 UI
                        "_type_score": type_score,
                    }
                    if any(word in name for word in whitelist):
                        hospitals.append(item)
                        if type_score >= 3:
                            high_quality_count += 1
                    else:
                        soft_candidates.append(item)
            except Exception:
                print(f"[search_nearby_hospitals] keyword={kw} 请求异常")
                print(traceback.format_exc())
                continue

            # 轻量提前停止：单关键词处理完毕后再判断，避免在关键词中途截断造成统计不一致
            if len(hospitals) >= HARD_COUNT_LIMIT or high_quality_count >= HIGH_QUALITY_THRESHOLD:
                print(
                    f"[search_nearby_hospitals] 提前停止 keyword={kw} "
                    f"collected={len(hospitals)} high_quality={high_quality_count}"
                )
                break

        # 所有关键词都失败时才返回错误提示；部分成功则返回成功结果
        if not any_keyword_succeeded:
            return [{"name": "❌ 高德接口全部请求失败", "address": "", "distance": "-", "tel": "-", "location": ""}]

        # fallback：strict whitelist 无结果时，回退到医疗专有词命中的软候选。
        # 不使用单字 "室/站/所/医" —— 会误匹配加油站(站)/派出所(所)/办公室(室)。
        # 真正的医疗"室/站/所"（卫生室/医务室/急救站）已包含 whitelist 中的
        # "卫生/医务/急救"，不会进 soft_candidates；这里补 whitelist 未覆盖的
        # 眼科/口腔/康复/体检/药房/疗养/妇产/心理/精神等医疗专有词。
        if not hospitals and soft_candidates:
            medical_fallback_kw = [
                "药房", "药店", "疗养", "康复", "体检",
                "口腔", "眼科", "妇产", "精神", "心理",
            ]
            for item in soft_candidates:
                nm = item["name"]
                if any(k in nm for k in medical_fallback_kw):
                    hospitals.append(item)
                if len(hospitals) >= 3:
                    break

        if not hospitals:
            return [{"name": "🔍 附近暂未找到正规医疗机构", "address": "", "distance": "-", "tel": "-", "location": ""}]

        # 限制最多 50 个候选：按 POI 距离取最近的 50 个，避免后续路线规划过多
        hospitals.sort(key=lambda x: _parse_distance(x["distance"]))
        return hospitals[:50]
    except Exception as e:
        print("[search_nearby_hospitals] 网络请求异常")
        print(traceback.format_exc())
        return [{"name": "❌ 网络请求错误", "address": str(e), "distance": "-", "tel": "-", "location": ""}]