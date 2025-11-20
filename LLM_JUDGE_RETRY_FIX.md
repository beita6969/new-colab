# LLM Judge重试机制修复

## 修复时间
2025-11-20 下午

## 问题描述
LLM Judge调用GPT OSS 120B时，偶尔会返回空内容（NoneType），导致评分失败。
- 发生频率: 13/97次 ≈ 13%
- 原因: GPT OSS 120B服务偶尔返回None
- 影响: 这些样本默认fallback到5.0分（中性）

## 解决方案

### 策略: 重试1次 + Fallback

**实现逻辑** (src/reward_computer.py:174-199):

```python
for attempt in range(2):  # 0=首次, 1=重试
    response = self.llm_judge_client.chat.completions.create(...)

    content = response.choices[0].message.content
    if content is None:
        if attempt == 0:
            print("⚠️  LLM Judge首次返回空内容，重试中...")
            continue  # 重试
        else:
            print("⚠️  LLM Judge重试后仍返回空内容，fallback判定为False")
            return False

    # 成功获取内容
    result_text = content.strip()
    break
```

### 优势

1. **鲁棒性**: 自动处理临时网络故障或服务抖动
2. **准确性**: 重试可能挽救13%本该正确的判决
3. **性能**: 只在失败时重试，成功时无额外开销
4. **透明性**: 打印详细日志，便于监控

## 预期效果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| NoneType错误 | 13次 | 预计<5次 |
| LLM Judge成功率 | 87% | 预计>95% |
| 评分准确性 | 受影响 | 显著改善 |

## 是否需要重启训练？

**建议**: 不需要立即重启
- 当前训练正在Step 4/500，运行稳定
- 这是一个增强性修复，不是阻塞性bug
- 可以等当前训练自然停止或完成后应用
- 如果希望立即生效，可以热重启（保存checkpoint后重启）

## 监控指标

重启后关注：
1. "重试中..." 日志出现频率
2. "重试后仍返回空内容" 日志（应该很少）
3. LLM Judge整体成功率
